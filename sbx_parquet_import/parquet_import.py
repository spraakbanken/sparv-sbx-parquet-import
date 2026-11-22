"""Example of a custom annotator."""

import datetime
import json
import math
import unicodedata
from collections import defaultdict
from typing import Any

import pyarrow
from pyarrow import parquet

from sparv.api import (
    Config,
    Output,
    Source,
    SourceFilename,
    SourceStructure,
    SparvErrorMessage,
    Text,
    get_logger,
    importer,
    util,
)

logger = get_logger(__name__)


def _serialize_nested(value: Any) -> Any:
    """Serialize nested values (structs and lists) as JSON. Other values are returned unchanged.

    Values that JSON can't represent, like dates, are converted to strings.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    return value


def _json_default(value: Any) -> str:
    """Convert a value that JSON can't represent to a string, using ISO format for dates and times."""
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


@importer(
    "parquet import",
    file_extension="parquet",
    outputs=["text", "document", Config("sbx_parquet_import.rows")],
    text_annotation="text",
    config=[
        Config(
            "sbx_parquet_import.prefix",
            description="Optional prefix to add to annotation names.",
            datatype=str,
        ),
        Config(
            "sbx_parquet_import.text_row",
            default="text",
            description="Name of the row containing the text.",
            datatype=str,
        ),
        Config(
            "sbx_parquet_import.rows",
            default=[],
            description="List of rows in source file. Only needed for renaming, as everything is parsed whether listed or not.",
            datatype=list,
        ),
        Config(
            "sbx_parquet_import.skip",
            default=[],
            description="Rows to skip during import.",
            datatype=list,
        ),
        Config(
            "sbx_parquet_import.batch_size",
            default=10000,
            description="Size of the batches to process at a time.",
            datatype=int,
        ),
        Config(
            "sbx_parquet_import.keep_control_chars",
            default=False,
            description="Set to True if control characters should not be removed from the text.",
            datatype=bool,
        ),
        Config(
            "sbx_parquet_import.normalize",
            default="NFC",
            description="Normalize input using any of the following forms: 'NFC', 'NFKC', 'NFD', and 'NFKD'.",
            datatype=str,
            choices=("NFC", "NFKC", "NFD", "NFKD"),
        ),
    ],
)
def parse(
    source_file: SourceFilename = SourceFilename(),
    source_dir: Source = Source(),
    text_row: str = Config("sbx_parquet_import.text_row"),
    rows: list = Config("sbx_parquet_import.rows"),
    skip: list = Config("sbx_parquet_import.skip"),
    batch_size: int = Config("sbx_parquet_import.batch_size"),
    prefix: str | None = Config("sbx_parquet_import.prefix"),
    keep_control_chars: bool = Config("sbx_parquet_import.keep_control_chars"),
    normalize: str = Config("sbx_parquet_import.normalize"),
) -> None:
    """Parse parquet file as input to Sparv.

    Note: Since parquet is a columnar format, the term "column" refers to one data entry and the term "row" refers to
    one annotation.

    Args:
        source_file: The source filename.
        source_dir: The source directory.
        rows: The list of rows in source file. Only needed for renaming, as everything is
            imported unless specified in skip.
        skip: Rows to skip during import.
        text_row: The name of the row containing the text to be imported.
        batch_size: The number of rows to process at a time.
        prefix: Optional prefix for output annotation.
        keep_control_chars: Set to True to keep control characters in the text.
        normalize: Normalize input text using any of the following forms: 'NFC', 'NFKC', 'NFD', and 'NFKD'.
            'NFC' is used by default.

    Raises:
        SparvErrorMessage: If the parquet file cannot be parsed.

    """
    logger.debug(f"Parsing parquet file '{source_file}'")

    source_file_path = source_dir.get_path(source_file, ".parquet")
    try:
        parquet_file = parquet.ParquetFile(str(source_file_path))
    except (OSError, pyarrow.ArrowException) as e:
        raise SparvErrorMessage(f"Failed to parse parquet file '{source_file}'. {type(e).__name__}: {e}") from None

    # Make sure the specified text row exists in the parquet file
    row_names = parquet_file.schema_arrow.names
    logger.debug(f"Row names in parquet file '{source_file}': {row_names}")
    if text_row not in row_names:
        raise SparvErrorMessage(f"Text row '{text_row}' not found in parquet file '{source_file}'.")

    # Assemble a set of rows to import with their target names, skipping any specified in the skip list
    import_rows = set()
    for row, target_name in sorted(util.misc.parse_annotation_list(rows)):
        if row not in skip:
            import_rows.add((row, target_name))
    for row in row_names:
        if row not in skip and not any(row == c for c, _ in import_rows):
            import_rows.add((row, row))

    # Process the parquet file in batches
    start_index = 0
    texts = []  # List of text rows
    text_spans = []  # List of text spans for each row
    annotations = defaultdict(list)  # List of annotations to be written later
    total_batches = math.ceil(parquet_file.metadata.num_rows / batch_size)
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size), start=1):
        logger.debug(f"Processing batch {i} of {total_batches} with {len(batch)} rows")

        # Note: technically, iter_batches() returns rows, not columns, but we keep calling them columns for consistency.
        for column in batch.to_pylist():
            # Extract text from text column (1 column = 1 data entry)
            text = (column[text_row])
            if not keep_control_chars:
                text = util.misc.remove_control_characters(text)
            if normalize:
                text = unicodedata.normalize(normalize, text)
            texts.append(text)
            text_spans.append((start_index, start_index + len(text)))
            start_index += len(text) + 2  # +2 for the newline characters

            # Extract other rows and add them as annotations
            for row, target_name in import_rows:
                if row != text_row:
                    annotations.setdefault(target_name, []).append(_serialize_nested(column[row]))

    # Write the extracted text
    text = "\n\n".join(texts)
    Text(source_file).write(text)

    # Write the text spans
    text_annotation = f"{prefix}.text" if prefix else "text"
    Output(text_annotation, source_file=source_file).write(text_spans)
    source_structure = [text_annotation]

    # Write the annotations
    for annotation_name, values in annotations.items():
        annotation_name = f"{text_annotation}:{annotation_name}"
        Output(annotation_name, source_file=source_file).write(values)
        source_structure.append(annotation_name)

    # Make up a document root node enclosing all texts
    document_annotation = f"{prefix}.document" if prefix else "document"
    Output(document_annotation, source_file=source_file).write([(0, text_spans[-1][1])])
    source_structure.append(document_annotation)

    # Write SourceStructure for text, document, and annotations
    SourceStructure(source_file).write(source_structure)

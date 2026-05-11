import re


def clean_text(text: str) -> str:
    """
    Light preprocessing for PDF-extracted research paper text.
    Keeps formulas, symbols, section titles, and scientific terms.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page markers like: --- Page 1 ---
    text = re.sub(r"\n?\s*--- Page \d+ ---\s*\n?", "\n\n", text)

    # Remove common bioRxiv footer/license noise
    text = re.sub(
        r"\.CC-BY 4\.0 International license.*?bioRxiv preprint",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Fix hyphenated line breaks: varia-\ntional -> variational
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert single line breaks inside paragraphs into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
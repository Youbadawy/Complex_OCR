# Medical Report Processor

A comprehensive tool for extracting structured information from medical reports, with a focus on mammography and breast imaging reports.

## Overview

The Medical Report Processor is designed to extract critical information from medical reports, transforming unstructured text into structured data for analysis. It features robust extraction capabilities that combine rule-based methods with advanced NLP and optional LLM integration.

Key features:
- Extract BIRADS scores with fuzzy matching to handle typos and variations
- Normalize dates, exam types, and other medical terminology
- Cross-validate extracted fields for consistency
- Enhance extraction with LLM capabilities (when available)
- Comprehensive error handling and logging
- Command-line interface for batch processing

## Installation

### Prerequisites
- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - python-dateutil
  - pypdf2
  - requests (for LLM integration)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medical-report-processor.git
cd medical-report-processor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Optional: Set up LLM integration by creating an `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Command Line Interface

The main script `process_reports.py` provides a command-line interface:

```bash
# Process a single PDF file
python process_reports.py --file path/to/report.pdf --output results.csv

# Process all PDFs in a directory
python process_reports.py --directory path/to/reports/ --output results.csv

# Fix an existing dataframe of results
python process_reports.py --fix-dataframe previous_results.csv --raw-text-dir path/to/raw_texts/ --output fixed_results.csv

# Disable LLM enhancement (faster processing)
python process_reports.py --directory path/to/reports/ --no-llm --output results.csv
```

### Python API

You can also use the library programmatically:

```python
from src.document_processing.main_processor import process_single_report, process_directory

# Process a single report
result = process_single_report("path/to/report.pdf")

# Process a directory of reports
df = process_directory("path/to/reports/", "output.csv")

# Fix issues in an existing dataframe
from src.document_processing.main_processor import fix_dataframe
fixed_df = fix_dataframe(original_df, raw_texts_dict)
```

## Architecture

The Medical Report Processor consists of several components:

1. **PDF Processing** (`pdf_processor.py`): Extracts text from PDF files.

2. **Text Analysis** (`text_analysis.py`): Parses raw text to identify key sections and extract structured information.

3. **Report Validation** (`report_validation.py`): Validates and normalizes extracted data, resolves inconsistencies.

4. **LLM Integration** (`llm_extraction.py`): Optional component that enhances extraction using large language models.

5. **Main Processor** (`main_processor.py`): Orchestrates the complete extraction pipeline.

## Extracted Fields

The processor extracts the following fields from medical reports:

- **exam_date**: Date of the examination (normalized to YYYY-MM-DD format)
- **exam_type**: Type of examination (e.g., MAMMOGRAM, ULTRASOUND)
- **birads_score**: BI-RADS assessment category (e.g., BIRADS 0-6)
- **clinical_history**: Patient history information
- **findings**: Detailed findings from the examination
- **impression**: Overall impression and assessment
- **recommendation**: Follow-up recommendations
- **patient_name**: Patient name (when available)
- **provider_info**: Information about referring and interpreting providers

## BIRADS Extraction

The system can recognize various formats of BIRADS scores:

- Standard format: "BIRADS 4", "BI-RADS Category 3"
- ACR format: "ACR 2", "Category 4a"
- Typos: "BLRADS 3", "BYRADS 4b"
- Contextual: "Normal findings" (inferred as BIRADS 1), "Suspicious mass" (inferred as BIRADS 4)

## Testing

Run the test suite to verify the extraction functions:

```bash
python -m unittest discover src/tests
```

Or run specific test modules:

```bash
python src/tests/test_validation.py
python src/tests/test_integration.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
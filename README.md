# DCF Builder

A Python tool for automated Discounted Cash Flow (DCF) analysis with machine learning-based growth projections.

## Features

- Automated financial data fetching using Yahoo Finance API
- Machine learning-based growth projections using XGBoost
- Excel template integration for professional DCF analysis
- Support for custom growth rates and forecast periods
- Beat frequency analysis for more accurate projections

## Prerequisites

- Python 3.8 or higher
- Excel template file (Edit-DCFtemplate.xlsx)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/michael/DCF-Builder.git
cd DCF-Builder
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place the Excel template file (Edit-DCFtemplate.xlsx) in the project directory.

## Usage

Run the script with a stock ticker:

```bash
python DCF_builder_v9.py AAPL
```

Optional arguments:
- `--years`: Number of forecast years (default: 5)
- `--growth`: Default growth rate if insufficient data (default: 0.05)

Example:
```bash
python DCF_builder_v9.py AAPL --years 7 --growth 0.08
```

## Output

The script generates an Excel file named `{TICKER}_DCF.xlsx` containing:
- Current financial data
- ML-based growth projections
- DCF valuation analysis
- Upside/downside analysis

## Notes

- The script requires an Excel template file (Edit-DCFtemplate.xlsx) to function
- Make sure you have a stable internet connection for data fetching
- Some companies may have limited financial data available

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
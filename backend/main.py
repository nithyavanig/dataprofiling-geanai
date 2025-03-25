import pandas as pd
import pdfplumber
import re
import json
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


class RegulatoryRuleExtractor:
    def __init__(self):
        """
        Initialize the extractor with class methods
        """
        self.dataset_path = None
        self.pdf_path = None
        self.columns = []
        self.extracted_rules = {}

    def _load_columns(self, dataset_path: str) -> List[str]:
        """
        Load column names from the dataset

        :param dataset_path: Path to Excel or CSV file
        :return: List of column names
        """
        try:
            if dataset_path.endswith('.xlsx'):
                df = pd.read_excel(dataset_path)
            elif dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                raise ValueError("Unsupported file format. Use .xlsx or .csv")

            return list(df.columns)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error reading dataset: {str(e)}"
            )

    def extract_rules(self, dataset_path: str, pdf_path: str, fuzzy_match: bool = True, context_window: int = 250) -> Dict[str, List[str]]:
        """
        Extract rules from PDF for each column

        :param dataset_path: Path to dataset file
        :param pdf_path: Path to PDF regulatory document
        :param fuzzy_match: Use fuzzy matching for column names
        :param context_window: Number of characters around the match to extract
        :return: Dictionary of column names and their associated rules
        """
        # Store paths
        self.dataset_path = dataset_path
        self.pdf_path = pdf_path

        # Load columns
        self.columns = self._load_columns(dataset_path)
        print("loading columns...")
        rules = {}

        # Open PDF
        with pdfplumber.open(pdf_path) as pdf:
            for column in self.columns:
                column_rules = []

                # Search through all pages
                for page in pdf.pages:
                    text = page.extract_text()
                    print("extracted text...")
                    # Different matching strategies
                    if fuzzy_match:
                        # Fuzzy matching - looks for similar column names
                        matches = self._fuzzy_find(text, column)
                    else:
                        # Exact matching
                        matches = self._exact_find(text, column)

                    # Extract rules near matches
                    for match_start, match_end in matches:
                        # Extract a wider context around the match
                        start = max(0, match_start - context_window)
                        end = min(len(text), match_end + context_window)
                        context = text[start:end].strip()

                        # Clean and validate the rule
                        print("Clean and validate the rule")
                        cleaned_rule = self._clean_rule(context)
                        if cleaned_rule:
                            column_rules.append(cleaned_rule)

                # Remove duplicates and sort
                print("Remove duplicates and sort")
                column_rules = sorted(set(column_rules), key=len, reverse=True)
                rules[column] = column_rules

        self.extracted_rules = rules
        return rules

    def _clean_rule(self, rule: str) -> str:
        """
        Clean and validate extracted rule

        :param rule: Raw extracted rule text
        :return: Cleaned rule text
        """
        # Remove excessive whitespace
        rule = re.sub(r'\s+', ' ', rule).strip()

        # Remove PDF artifacts
        rule = re.sub(r'\n+', ' ', rule)

        # Minimum meaningful length
        if len(rule) < 20:
            return ''
        print("cleaned rule")
        return rule

    def _fuzzy_find(self, text: str, column: str) -> List[tuple]:
        """
        Fuzzy matching of column names in text

        :param text: Text to search
        :param column: Column name to match
        :return: List of match start and end indices
        """
        matches = []
        lower_text = text.lower()
        lower_column = column.lower()

        # Various fuzzy matching approaches
        patterns = [
            # Exact word match
            rf'\b{re.escape(lower_column)}\b',
            # Partial match
            rf'{re.escape(lower_column)}',
            # Variations with common separators
            rf'{re.escape(lower_column.replace(" ", "_"))}',
            rf'{re.escape(lower_column.replace(" ", "-"))}'
        ]

        # Try different patterns
        for pattern in patterns:
            for match in re.finditer(pattern, lower_text):
                matches.append((match.start(), match.end()))
        print("fuzzy find")
        return matches

    def _exact_find(self, text: str, column: str) -> List[tuple]:
        """
        Exact matching of column names in text

        :param text: Text to search
        :param column: Column name to match
        :return: List of match start and end indices
        """
        matches = []
        lower_text = text.lower()
        lower_column = column.lower()

        # Find all exact occurrences
        for match in re.finditer(rf'\b{re.escape(lower_column)}\b', lower_text):
            matches.append((match.start(), match.end()))
        print("exact find")
        return matches

    def refine_rules(self, rules: Dict[str, List[str]], column: str, actions: List[Dict]) -> Dict[str, List[str]]:
        """
        Refine rules for a specific column

        :param rules: Current rules dictionary
        :param column: Column to refine
        :param actions: List of rule modification actions
        :return: Updated rules dictionary
        """
        if column not in rules:
            raise HTTPException(status_code=404, detail=f"Column {column} not found")

        column_rules = rules[column]

        for action in actions:
            action_type = action.get('type')

            if action_type == 'remove':
                # Remove rule by index
                index = action.get('index')
                if 0 <= index < len(column_rules):
                    del column_rules[index]

            elif action_type == 'add':
                # Add new rule
                new_rule = action.get('rule')
                if new_rule:
                    column_rules.append(new_rule)

            elif action_type == 'edit':
                # Edit existing rule
                index = action.get('index')
                new_rule = action.get('rule')
                if 0 <= index < len(column_rules) and new_rule:
                    column_rules[index] = new_rule

        # Update rules
        rules[column] = column_rules
        return rules


# FastAPI Application
app = FastAPI(
    title="Regulatory Rule Extraction Service",
    description="API for extracting and refining regulatory rules from datasets and PDFs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global extractor instance
extractor = RegulatoryRuleExtractor()
RESOURCES_DIR = "resources"


@app.post("/extract-rules/")
async def extract_rules(
    # dataset: UploadFile = File(...),
    # pdf: UploadFile = File(...),
    # fuzzy_match: bool = True,
    # context_window: int = 250
):
    """
    Extract regulatory rules from uploaded dataset and PDF

    :param dataset: Uploaded dataset file (.xlsx or .csv)
    :param pdf: Uploaded PDF regulatory document
    :param fuzzy_match: Enable fuzzy matching
    :param context_window: Context window for rule extraction
    :return: Extracted rules
    """
    # Save uploaded files
    # dataset_path = f"temp_{dataset.filename}"
    # pdf_path = f"temp_{pdf.filename}"
    
    fuzzy_match: bool = True,
    context_window: int = 250
    
    current_dir = os.getcwd()
    os.path.join(current_dir, "resources")
    dataset_path = os.path.join(RESOURCES_DIR, "input_dataset.xlsx")
    pdf_path = os.path.join(RESOURCES_DIR, "federal_reserve_regulations.pdf")

    try:
        # Save dataset file
        # with open(dataset_path, "wb") as buffer:
        #     buffer.write(await dataset_path.read())

        # # Save PDF file
        # with open(pdf_path, "wb") as buffer:
        #     buffer.write(await pdf_path.read())

        # Extract rules
        rules = extractor.extract_rules(
            dataset_path,
            pdf_path,
            fuzzy_match=fuzzy_match,
            context_window=context_window
        )

        return JSONResponse(content=rules)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        print("extracted rules..")
        # Clean up temporary files
        # if os.path.exists(dataset_path):
        #     os.remove(dataset_path)
        # if os.path.exists(pdf_path):
        #     os.remove(pdf_path)


@app.post("/refine-rules/")
async def refine_rules(
    column: str = Body(...),
    rules: Dict[str, List[str]] = Body(...),
    actions: List[Dict] = Body(...)
):
    """
    Refine rules for a specific column

    :param column: Column to refine
    :param rules: Current rules dictionary
    :param actions: List of rule modification actions
    :return: Updated rules
    """
    try:
        refined_rules = extractor.refine_rules(rules, column, actions)
        return JSONResponse(content=refined_rules)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export-rules/")
async def export_rules(
    rules: Dict[str, List[str]] = Body(...),
    format: str = "json"
):
    """
    Export refined rules to a file

    :param rules: Rules to export
    :param format: Export format (json or markdown)
    :return: Exported file
    """
    try:
        if format == "json":
            # Export as JSON
            output_path = "regulatory_rules.json"
            with open(output_path, "w") as f:
                json.dump(rules, f, indent=2)
            return FileResponse(output_path, media_type="application/json", filename="regulatory_rules.json")

        elif format == "markdown":
            # Export as Markdown
            output_path = "regulatory_rules_report.md"
            with open(output_path, "w", encoding='utf-8') as f:
                f.write("# Regulatory Rules Analysis Report\n\n")
                for column, column_rules in rules.items():
                    f.write(f"## Column: {column}\n\n")

                    if not column_rules:
                        f.write("*No rules found or retained for this column.*\n\n")
                    else:
                        f.write("### Refined Regulatory Rules:\n")
                        for i, rule in enumerate(column_rules, 1):
                            f.write(f"{i}. {rule}\n")
                        f.write("\n")

            return FileResponse(output_path, media_type="text/markdown", filename="regulatory_rules_report.md")

        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
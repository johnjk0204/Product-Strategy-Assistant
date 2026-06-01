import csv
import io
import json
from typing import List
import pandas as pd


class DocumentProcessor:
    """Process various document types into text chunks for embedding."""

    def process(self, filename: str, content: bytes) -> List[str]:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "csv":
            return self._process_csv(content)
        elif ext == "json":
            return self._process_json(content)
        elif ext == "pdf":
            return self._process_pdf(content)
        else:
            return self._process_text(content)

    # ------------------------------------------------------------------
    def _process_csv(self, content: bytes) -> List[str]:
        df = pd.read_csv(io.BytesIO(content))
        chunks: List[str] = []

        # Dataset overview
        overview = (
            f"Dataset Overview:\n"
            f"Total rows: {len(df)}\n"
            f"Columns: {', '.join(df.columns.tolist())}\n"
        )
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols):
            overview += "\nNumeric Statistics:\n"
            for col in numeric_cols:
                overview += (
                    f"  {col}: min={df[col].min():.2f}, "
                    f"max={df[col].max():.2f}, "
                    f"avg={df[col].mean():.2f}, "
                    f"total={df[col].sum():.2f}\n"
                )
        chunks.append(overview)

        # Category distributions
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() <= 20:
                cat_text = f"\n{col} Distribution:\n"
                for val, cnt in df[col].value_counts().items():
                    cat_text += f"  {val}: {cnt} records\n"
                chunks.append(cat_text)

        # Product performance summary
        if "Product_Name" in df.columns and "Revenue_USD" in df.columns:
            agg_cols = {"Revenue_USD": "sum"}
            if "Units_Sold" in df.columns:
                agg_cols["Units_Sold"] = "sum"
            if "Profit_USD" in df.columns:
                agg_cols["Profit_USD"] = "sum"
            if "Customer_Rating" in df.columns:
                agg_cols["Customer_Rating"] = "mean"
            if "Returns" in df.columns:
                agg_cols["Returns"] = "sum"
            if "Marketing_Spend_USD" in df.columns:
                agg_cols["Marketing_Spend_USD"] = "sum"

            product_perf = df.groupby("Product_Name").agg(agg_cols).round(2)
            text = "Product Performance Summary:\n"
            for product, row in product_perf.iterrows():
                text += f"\n{product}:\n"
                for metric, value in row.items():
                    text += f"  {metric}: {value:,.2f}\n"
            chunks.append(text)

        # Regional performance
        if "Region" in df.columns and "Revenue_USD" in df.columns:
            region_agg = {"Revenue_USD": "sum"}
            if "Units_Sold" in df.columns:
                region_agg["Units_Sold"] = "sum"
            if "Profit_USD" in df.columns:
                region_agg["Profit_USD"] = "sum"
            region_perf = df.groupby("Region").agg(region_agg).round(2)
            text = "Regional Performance Summary:\n"
            for region, row in region_perf.iterrows():
                text += f"\n{region} Region:\n"
                for metric, value in row.items():
                    text += f"  {metric}: {value:,.2f}\n"
            chunks.append(text)

        # Monthly trends
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols and "Revenue_USD" in df.columns:
            df["_date"] = pd.to_datetime(df[date_cols[0]])
            df["_month"] = df["_date"].dt.to_period("M").astype(str)
            monthly = df.groupby("_month")["Revenue_USD"].sum().round(2)
            text = "Monthly Revenue Trends:\n"
            for month, rev in monthly.items():
                text += f"  {month}: ${rev:,.2f}\n"
            chunks.append(text)

        # Customer reviews in batches of 15
        review_cols = [
            c for c in df.columns
            if any(w in c.lower() for w in ["review", "comment", "feedback"])
        ]
        for col in review_cols:
            reviews = df[col].dropna().tolist()
            for i in range(0, len(reviews), 15):
                batch = reviews[i: i + 15]
                text = f"Customer Reviews (batch {i // 15 + 1}):\n"
                for j, r in enumerate(batch):
                    text += f"  {i + j + 1}. {r}\n"
                chunks.append(text)

        # Rating distribution
        if "Customer_Rating" in df.columns:
            rating_dist = df["Customer_Rating"].value_counts().sort_index()
            text = "Customer Rating Distribution:\n"
            for rating, count in rating_dist.items():
                text += f"  {rating} stars: {count} records\n"
            avg = df["Customer_Rating"].mean()
            text += f"  Average Rating: {avg:.2f}\n"
            chunks.append(text)

        return chunks

    def _process_text(self, content: bytes) -> List[str]:
        text = content.decode("utf-8", errors="ignore")
        words = text.split()
        chunks = [" ".join(words[i: i + 800]) for i in range(0, len(words), 800)]
        return [c for c in chunks if c.strip()] or [text[:4000]]

    def _process_json(self, content: bytes) -> List[str]:
        try:
            data = json.loads(content.decode("utf-8"))
            return self._process_text(json.dumps(data, indent=2).encode())
        except Exception:
            return self._process_text(content)

    def _process_pdf(self, content: bytes) -> List[str]:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "".join(p.extract_text() or "" for p in pdf.pages)
            return self._process_text(text.encode())
        except ImportError:
            pass
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "".join(p.extract_text() or "" for p in reader.pages)
            return self._process_text(text.encode())
        except Exception:
            return ["Unable to extract PDF text. Please convert to .txt and re-upload."]

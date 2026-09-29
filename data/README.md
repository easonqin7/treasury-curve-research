# Raw yield data

The versioned CSV contains unfilled daily FRED Treasury constant-maturity observations. Columns are FRED series IDs; yields are in percent. Missing observations are empty fields, not zero yields.

Source attribution: Board of Governors of the Federal Reserve System (US), H.15 Selected Interest Rates, retrieved from FRED, Federal Reserve Bank of St. Louis. Series IDs and retrieval details appear in `fred_yields_raw.metadata.json`. Example series and source notes: https://fred.stlouisfed.org/series/DGS10.

The series page identifies the data as public domain with citation requested. This attribution does not extend to unrelated third-party FRED series. The data file is not individual bond prices or an execution dataset.

The extra leading all-missing 2010-01-01 row is preserved. It is dropped by the legacy baseline before its first usable 2010-01-04 observation. Do not infer holidays by deleting unchanged quotes.

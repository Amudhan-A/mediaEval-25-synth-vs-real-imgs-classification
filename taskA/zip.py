import zipfile

with zipfile.ZipFile("bitStop_open.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("bitStop_open.csv")

print("CSV zipped successfully!")
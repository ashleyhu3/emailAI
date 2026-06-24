from utils import get_file_hash
from pathlib import Path

file_path = Path("/Users/davidfu/Desktop/Rays_Intern/PDF_summarizer/research_pdfs/1_91APP (6741) - HK Roadshow - Nov 7-8th, 2024_91App HK Road Nov 7-8.pdf")  # pick any small PDF

print("hi")
hash1 = get_file_hash(file_path)
hash2 = get_file_hash(file_path)

print(f"Hash 1: {hash1}")
print(f"Hash 2: {hash2}")
print("Hashes match?", hash1 == hash2)

# Optional: modify the file slightly and check hash changes
# with open(file_path, "ab") as f:
#     f.write(b"extra bytes")
# hash3 = get_file_hash(file_path)
# print("Modified hash:", hash3)
# print("Hash changed?", hash1 != hash3)

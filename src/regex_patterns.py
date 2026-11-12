import re

text = """
This dress is beautiful!!!
Price is $45.
Visit https://example.com
Email: test@gmail.com
"""

print("Original:")
print(text)

print("\nRemove URLs:")
print(re.sub(r"https?://\S+", "", text))

print("\nRemove emails:")
print(re.sub(r"\S+@\S+", "", text))

print("\nRemove numbers:")
print(re.sub(r"\d+", "", text))

print("\nRemove punctuation:")
print(re.sub(r"[^\w\s]", "", text))

print("\nRemove extra spaces:")
print(re.sub(r"\s+", " ", text).strip())
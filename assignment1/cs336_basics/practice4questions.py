# Practice 4 Questions
# BPE 2.1 The unicode Standard

c = chr(0)

# question (a)
print(c)

# question (b)
print(repr(c))

# question (c)
s = "this is a test" + chr(0) + "string"
print(s)

# BPE 2.2 The Unicode Encoding
# question (b)
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

s1 = "你"
b = s1.encode("utf-8")

# print(b)
# print(decode_utf8_bytes_to_str_wrong(b))
"""
Information and Probability(Claude Shannon)
===================================
Information I(x) and Probability P(x) are inversely related.

Self Information: I(x) = logb(1/P(x)) = -logbP(x)

High Probability Event -> Low Information
Low Probability Event -> High Information

For binary, b=2
i.e. I(x) = log2(1/P(x)), so we allocate fewer bits to high probability symbols in the text

"""

import math
import time
import sys


class Node:
    def __init__(self, freq, char=None, left=None, right=None):
        self.char = char
        self.freq = freq
        self.binary = None
        self.left = left
        self.right = right

    def __str__(self):
        _str = ""
        if self.char:
            _str += self.char + ": "
        _str += f"{self.freq}"
        if self.binary:
            _str += f" Binary: {self.binary}"
        if self.left:
            _str += f" Left: {self.left}"
        if self.right:
            _str += f" Right: {self.right}"
        return _str

    def is_leaf(self):
        if self.right and self.left:
            return False
        return True


class Tree:
    def __init__(self, root: Node):
        self.root = root

    def label(self, node: Node):
        if node.is_leaf():
            return 0
        if node == self.root:
            node.binary = ""
        node.left.binary = node.binary + "0"
        node.right.binary = node.binary + "1"
        return self.label(node.left) + self.label(node.right)

    def encoding_dict(self, node: Node):
        enc_dict = {}
        if node.char:
            enc_dict[node.char] = node.binary
        if node.is_leaf():
            return enc_dict
        return self.encoding_dict(node.left) | self.encoding_dict(node.right)


"""
Huffman Tree 
=========================================================================
1.) Intialize n one-node trees labeled with the characters of the alphabet
2.) Record the frequency/weight of the character in the root
3.) Repeat until a single tree is obtained:
    * Find two trees with the smallest weight
    * Make them the left and right sub-tree of a new tree and
    record the sum of their weights in the root

4.)After we have generated the huffman tree we start with current node set to the root
5.)If node is not a leaf node, label the edge to the left child as 0 and the edge to the 
right child as 1. Repeat the process at both the left and right child.
6.)Encoding of any symbol is then read by a concatenation of the labels on the
edges along the path from the root node to the symbol

***prefix freeness



 """


def huffman(text: str):
    freq_dict = {}
    for chr in text:
        if chr not in freq_dict:
            freq_dict[chr] = 1
        else:
            freq_dict[chr] += 1
    # print(f"{freq_dict}\n\n")

    zipped_chrs = list(zip(freq_dict.keys(), freq_dict.values()))
    zipped_chrs.sort(key=lambda x: x[1])
    trees = [Node(x[1], x[0]) for x in zipped_chrs]
    while True:
        if len(trees) == 1:
            break
        left_subtree = trees[0]
        right_subtree = trees[1]
        trees.pop(0)
        trees.pop(0)
        tree = Node(
            left_subtree.freq + right_subtree.freq,
            None,
            left_subtree,
            right_subtree,
        )
        # print(tree)
        trees.append(tree)
        trees.sort(key=lambda x: x.freq)

    tree = Tree(trees[0])
    tree.label(tree.root)
    enc_dict = tree.encoding_dict(tree.root)
    return enc_dict


"""
Canonical Encoding
=============================================================

Now that we have a huffman tree we can compress our symbolic data.
Decompression is an issue, storing the tree in the compressed file is expensive.

We use Canonical Huffman Encoding. The important info is the mapping from symbols to
codeword lengths, the actual bit pattern is less important. 
We transofrm the existing codes:

1.) The first symbol in list gets assigned a codeword which is the same length as the symbol's
codeword but all zeroes.
2.) Each subsequent symbol is assigned the next binary number in the sequence ensuring 
that following codes are always higher in value.
3.) When you reach a longer codeword, then after incrementing, append zeroes
until the length of the new codeword is equal to the length of the old codeword(left shift).


Lookup Table
=================================================================

We build a lookup table to speed up decoding. 
(Much faster than traversing tree or doing string matching) 
It allows us to evaluate fixed sized chunks of the compressed
text very quickly.

Suppose our maximum length code is 16 bits.
We have to take in 16 bits at a time, therefore our lookup table needs to 
map 2^16 values.

For each index in the table we transform it into its 16 bit representation,
if that 16 bit number begins with a particular code, then we associate it with that char's
code and the number of bits in that code.

eg. Suppose 'a' is 1101 and we have 16 bit max.

Then an entry like 1101000000000000 will be associated with 'a', and its num bits set to 4.
So will 1101000000000010.

When decoding we work with a buffer of 16 bits. Turn them into an int, use the lookup table for char and then
'consume' the number of bits specified by the lookup table.
"""


def canonical_encoding(enc_dict):
    symbol_pairs = list(zip(enc_dict.keys(), enc_dict.values()))
    symbol_pairs = sorted(
        map(lambda x: [x[0], len(x[1])], symbol_pairs), key=lambda x: x[1]
    )

    for i in range(0, len(symbol_pairs)):
        if i == 0:
            symbol_pairs[0][1] = "0" * symbol_pairs[0][1]
        else:
            temp = f"{(int(symbol_pairs[i - 1][1], 2) + 1):b}"
            temp = temp.rjust(len(symbol_pairs[i - 1][1]), "0")
            symbol_pairs[i][1] = temp.ljust(symbol_pairs[i][1], "0")

    enc_dict = dict(symbol_pairs)

    dec_dict = dict(
        sorted(
            map(lambda x: [x[1], x[0]], symbol_pairs),
            key=lambda x: len(x[0]),
            reverse=True,
        )
    )

    max_size = len(next(iter(dec_dict)))
    # print(max_size)
    lookup_table = [None] * (2**max_size)
    dict_list = list(
        (sorted(dec_dict.items(), key=lambda x: int(x[0].rjust(max_size, "0"), 2)))
    )
    i = 0
    for j in range(0, (2**max_size)):
        bin_str_rep = bin(j)[2:].rjust(max_size, "0")
        if not bin_str_rep.startswith(dict_list[i][0]):
            i += 1
        lookup_table[j] = (dict_list[i][1], len(dict_list[i][0]))

    return enc_dict, lookup_table


"""
Compression and File Header
=======================================

Compress using canonical encoding and provide enough information to be able to decode.
All we need is the characters in our file's alphabet and the number of bits allocated to each character in order to
rebuild the canonical encoding.

I'm sure better header formats exist but this seems fairly reasonable to me:

According to Wikipedia UTF-8 is capable of encoding 1 112 064 symbols, 21 bits gives us up to 2 097 151.(2^21 -1)
[21 bits for number of symbols(N)]<N times>[n bits where 2^n -1 is the size of the alphabet(N), containing num bits of char]
[UTF-8 encoding of char]</>[1 byte for amount of padding]

(So N is the size of our alphabet)

"""


def compress(text: str, enc_dict: dict, filename: str):
    compressed = ""
    alphabet_size = f"{len(enc_dict):b}".rjust(21, "0")
    compressed += alphabet_size

    for char, code in enc_dict.items():
        num_bits = f"{len(code):b}".rjust(math.ceil(math.log2(len(enc_dict) + 1)), "0")
        compressed += num_bits
        char_rep = ""
        for _bin in char.encode():
            char_rep += f"{_bin:b}".rjust(8, "0")

        compressed += char_rep

    for char in text:
        compressed += enc_dict[char]
    with open(filename, "wb") as f:
        unpadded_length = len(compressed)
        if unpadded_length % 8 != 0:
            compressed.ljust(unpadded_length + unpadded_length % 8, "0")
            padding_info = f"{unpadded_length%8:b}".rjust(8, "0")
            compressed += padding_info
        byte_array = bytearray()
        for i in range(0, len(compressed), 8):
            byte_array.append(int(compressed[i : i + 8], 2))
        f.write(byte_array)


def make_binary_arr(_bytes):
    full_binary_arr = []

    for x in _bytes:
        bin_arr = [0, 0, 0, 0, 0, 0, 0, 0]
        while x > 0:
            i = math.floor(math.log2(x))
            x = x - (2**i)
            if x >= 0:
                bin_arr[8 - i - 1] = 1
        full_binary_arr.extend(bin_arr)
    return full_binary_arr


def bin_arr_to_int(bin_arr):
    out = 0
    for bit in bin_arr:
        out = (out << 1) | bit
    return out


"""
Decompression
====================
The first 21 bits tell us how big our alphabet is.
Last byte tells us how much padding to remove.

For each symbol we have math.ceil(math.log2(size_alpha + 1)) bits which tell us 
how many bits are allocated to that char and then the utf-8 for that char.
UTF-8 characters range from 1 to 4 bytes, the format of the first byte tells us how many bytes 
our character consists of.


We then recreate the canonical encoding.
Then we decode using the lookup table.

"""


def decompress(file: str, out_file: str):
    with open(file, "rb") as f:
        binary_content = f.read()
        binary_str = ""
    bin_arr = make_binary_arr(binary_content)
    for byte in binary_content:
        binary_str += f"{byte:b}".rjust(8, "0")

    amount_of_padding = bin_arr_to_int(bin_arr[-8:])
    bin_arr = bin_arr[: len(binary_str) - (8 + amount_of_padding)]
    binary_str = binary_str[: len(binary_str) - (8 + amount_of_padding)]
    size_alpha = bin_arr_to_int(bin_arr[:21])
    bits_for_length = math.ceil(math.log2(size_alpha + 1))
    i = 21
    num_chars = 0
    char_lengths_dict = {}
    while True:
        if num_chars == size_alpha:
            break
        num_bits_of_char = bin_arr_to_int(bin_arr[i : i + bits_for_length])
        i += bits_for_length
        first_utf_byte = bin_arr[i : i + 8]
        second_candidate_byte = bin_arr[i + 8 : i + 16]
        third_candidate_byte = bin_arr[i + 16 : i + 24]
        fourth_candidate_byte = bin_arr[i + 24 : i + 32]
        if first_utf_byte[0] == 0:
            char = chr(bin_arr_to_int(first_utf_byte))
            char_lengths_dict[char] = num_bits_of_char * "0"
            i += 8
            num_chars += 1
        elif first_utf_byte[:3] == [1, 1, 0]:
            full_utf = [
                bin_arr_to_int(first_utf_byte),
                bin_arr_to_int(second_candidate_byte),
            ]
            char = bytearray(full_utf).decode("utf-8")
            char_lengths_dict[char] = num_bits_of_char * "0"
            i += 16
            num_chars += 1
        elif first_utf_byte[:4] == [1, 1, 1, 0]:
            full_utf = [
                bin_arr_to_int(first_utf_byte),
                bin_arr_to_int(second_candidate_byte),
                bin_arr_to_int(third_candidate_byte),
            ]
            char = bytearray(full_utf).decode("utf-8")
            char_lengths_dict[char] = num_bits_of_char * "0"
            i += 24
            num_chars += 1
        elif first_utf_byte[:5] == [1, 1, 1, 1, 0]:
            full_utf = [
                bin_arr_to_int(first_utf_byte),
                bin_arr_to_int(second_candidate_byte),
                bin_arr_to_int(third_candidate_byte),
                bin_arr_to_int(fourth_candidate_byte),
            ]
            char = bytearray(full_utf).decode("utf-8")
            char_lengths_dict[char] = num_bits_of_char * "0"
            i += 36
            num_chars += 1
    enc_dict, lookup_table = canonical_encoding(char_lengths_dict)
    compressed = binary_str[i:]
    decode(compressed, lookup_table, out_file)


"""
We use Lookup Table and move over the compressed string.


Windows(\r\n) vs Linux(\n) style newlines can lead to different file sizes to original with same content.
"""


def decode(compressed, lookup_table: list, decomp_file: str):
    size_compressed = len(compressed)
    decompressed = ""
    BUFF_SIZE = int(math.log2(len(lookup_table)))
    buff = ""
    i = 0

    while True:
        if i >= size_compressed:
            break
        chars_to_add = int(BUFF_SIZE - len(buff))
        if i + chars_to_add > size_compressed:
            # print("here")
            break
        buff += compressed[i : chars_to_add + i]

        i += chars_to_add
        char, bit_length = lookup_table[int(buff, 2)]
        decompressed += char
        buff = buff[bit_length:]

    with open(decomp_file, "w") as f:
        f.writelines(decompressed + "\n")
        # We add a newline because unix style files have \n at end


"""
Sanity check
"""


def validate(input_file: str, output_file: str):
    with open(input_file, "r") as f:
        in_text = f.readlines()

    with open(output_file, "r") as f:
        out_text = f.readlines()

    in_length = len(in_text)
    out_length = len(out_text)
    if in_length != out_length:
        print(f"Original has {in_length} lines vs Decompressed {out_length} lines.")
        if in_length < out_length:
            print(f"Excess lines:")
            for i in range(in_length, out_length):
                print(f"{i+1}:{out_text[i]}")
        else:
            print(f"Missing lines:")
            for i in range(out_length, in_length):
                print(f"{i+1}:{in_text[i]}")

    line_num = 1
    for line1, line2 in zip(in_text, out_text):
        if line1 != line2:
            print(f"Does not match on {line_num}: '{line1}' vs '{line2}'")
        line_num += 1


def main():

    u_args = sys.argv[1:]
    # print(u_args)

    HELP_TEXT = (
        f"Usage of huffman:\n"
        f"Compress: -c <input file> <output binary file>\n"
        f"Decompress: -d <input file> <output text file>\n"
        f"Validate: -v <original> <decompressed>\nHelp: -h"
    )
    # print(u_args)

    if len(u_args) != 3:
        print(HELP_TEXT)

    elif u_args[0] == "-c":
        print("Compressing")
        with open(u_args[1], "r") as f:
            text = f.readlines()
        text = "".join(text)
        comp_delta = time.time()
        enc_dict = huffman(text)
        enc_dict, lookup_table = canonical_encoding(enc_dict)
        compress(text, enc_dict, u_args[2])
        print(f"Compressed in: {time.time() - comp_delta}s")

    elif u_args[0] == "-d":
        print("Decompressing")
        decomp_delta = time.time()
        decompress(u_args[1], u_args[2])
        print(f"Decompressed in: {time.time()-decomp_delta}")

    elif u_args[0] == "-v":
        print("Validating")
        validate(u_args[1], u_args[2])
        print("Done!")
    elif u_args[0] == "-h":
        print(HELP_TEXT)
    else:
        print(HELP_TEXT)


if __name__ == "__main__":
    main()

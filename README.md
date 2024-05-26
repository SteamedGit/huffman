# Huffman Encoding in Python
This is an implementation of Huffman Encoding written in Python 3 from scratch. It makes use of techniques like Canonical Encoding and Lookup Tables in order to make file headers smaller and decoding faster.

* [More on Huffman Encoding](https://en.wikipedia.org/wiki/Huffman_coding)
* [More on Canonical Huffman Encoding](https://en.wikipedia.org/wiki/Canonical_Huffman_code)
* [More on Lookup Tables](https://en.wikipedia.org/wiki/Lookup_table)

Note that the code also explains each step of the process.

## Usage
* Compress: -c input file output binary file
* Decompress: -d input file output text file
* Validate: -v original decompressed

## Performance
The code was mainly tested on files from the [Cantebury Corpus](https://corpus.canterbury.ac.nz/descriptions/), specifically the original 1997 version and the Large Corpus.

The bible provided in the Large Corpus is compressed from 3 953KB to 2 167KB.
Alice in Wonderland is compressed from from 149KB to 83KB.


## Other considerations
Due to different newline endings on Linux(\n) and Windows(\r\n) decompressed files may be a different size to the original despite their content being identical.


### Example Usage

```
python huffman.py -c bible.txt compressed
python huffman.py -d compressed decompressed.txt
python huffman.py -v bible.txt decompressed.txt
```





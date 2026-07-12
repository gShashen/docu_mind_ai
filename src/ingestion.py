def exctract_text_from_pdf(file):

    """
    Extracts text from each page of a PDF and returns it as a list of dicts.

    Args:
        file (pymupdf.Document): An opened PyMuPDF document object.

    Returns:
        list: A list of dicts with keys 'page' (int) and 'text' (str).
    """

    pages = []

    for page in file:

        page_info = {}            
        text = page.get_text()

        page_info['page'] = page.number + 1
        page_info['text'] = text

        pages.append(page_info)

    return pages

def chunk_text(pages,chunk_size=100,overlap=20):

    """
    Extract chunks from each page and returns it a list of dicts

    Args:
        pages (list): A list of dicts with keys 'page' (int) and 'text' (str).
        chunk_size (int): Number of words per chunk. Defaults to 100.
        overlap (int): Number of overlapping words between chunks. Defaults to 20.

    Returns:
        A list of dicts with keys 'chunk_id' (int), 'page' (int) and 'text' (str)
    """

    chunks = []
    chunk_id = 0

    for page in pages:
        words = page['text'].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i: i+ chunk_size]
            chunk_text = " ".join(chunk_words) 
            
            chunks.append(
                {
                    "chunk_id" : chunk_id,
                    "page" : page['page'],
                    "text": chunk_text
                }
            )

            chunk_id +=1
    return chunks

if __name__ == "__main__":
    pages = exctract_text_from_pdf("data/test.pdf")
    print(f"Pages extracted: {len(pages)}")

    chunks = chunk_text(pages)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk preview:\n{chunks[0]}")
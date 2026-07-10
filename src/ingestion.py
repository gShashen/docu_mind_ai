import pymupdf

def exctract_text_from_pdf(pdf_path):

    pages = []

    with pymupdf.open(pdf_path) as file:
        for page in file:

            page_info = {}            
            text = page.get_text()

            page_info['page'] = page.number + 1
            page_info['text'] = text

            pages.append(page_info)

    return pages


def chunk_text(pages,chunk_size=500,overlap=50):

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
    
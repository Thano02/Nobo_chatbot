import pickle
from sentence_transformers import SentenceTransformer
import re
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import gspread

# === CONFIGURATION ===
SERVICE_ACCOUNT_FILE = "chatbot-camp-4d9be0433c1f.json"  # Mets le vrai nom ici
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly"
]

google_docs_urls =  [
    "https://docs.google.com/document/d/1hJSHcPFrKjV-jRuCvoxnpulpM98TNO3mSt7lUpb-t54/edit",
    "https://docs.google.com/document/d/1ZwKk8dG7fRM0gzUYi83Of93CZAg7Fn8u8n8TwZBUrK4/edit",
    "https://docs.google.com/document/d/15trRtqzbHLXyp3ddRAq2A4ru-7E_oh4r82IftUH6T9I/edit",
    "https://docs.google.com/document/d/1IBofMnjb_FmlfqGOYZjOvbRVbEcDFZuxrzQnzudVCNE/edit",
    "https://docs.google.com/document/d/1JOSRs0pwCbFxsewsYK3oXiIiJsqgbQQAJieCe91rZA0/edit",
    "https://docs.google.com/document/d/1Nz0g7rnel2DGmPK6RZdGtRlLydc0Z5XCqXRGa92CIR4/edit",
    "https://docs.google.com/document/d/1I_dN8CTGsGTpsQqH0gnDbal5Wfz_ZnBqZG69G7G11d0/edit",
    "https://docs.google.com/document/d/1xprfNO2Zr2EMSl4GaZNwZbbI5IoHUh2lUtv-bbtKn8Q/edit",
    "https://docs.google.com/document/d/1Y8glPs5S0YH4KnEQ7szTlFQ9DB21LYtYc_vng2PbKaA/edit",
    "https://docs.google.com/document/d/157tzzeVkqzvViWfedpFQkcpF3pvmbxmofMUZejn-sRk/edit",
    "https://docs.google.com/document/d/11dNOa49n1L79H67Yl1XC1o3Wd929qPFNmkinlwSxJHM/edit",
    "https://docs.google.com/document/d/1zJFsdtVvNNbFmTS1DktCjoyVtf0T1Wr3-tXW27infg0/edit",
    "https://docs.google.com/document/d/1fBKVKuGpo7dEI_03csvoIePYEHbv6DkC3tp4ADFRDvQ/edit?tab=t.0#heading=h.1cdh11chfqhy"
]

google_sheets_urls =  [
    "https://docs.google.com/spreadsheets/d/1wzrXJPZTrN1PRSqBy9aQloySxzRgrwOT_t_8xELUgrg/edit",
    #"https://docs.google.com/spreadsheets/d/1m4gXVV9Ce1daP5jJya9jMdw17IMw8SRY/edit" excel pour le moment,
    "https://docs.google.com/spreadsheets/d/1qKGTC8mP-_UOjquuxiWu4J0BheaW6KgEMZ7ma3d-kYg/edit",
    "https://docs.google.com/spreadsheets/d/11RGnu-QU1PxOHZAQK1pUMDC5MB_BvAsDPSjkVH_5-e8/edit",
    "https://docs.google.com/spreadsheets/d/1_njJyDuDmJ-WpXQ_nMNB3xmp7uE9vomTuPe2zyseino/edit",
    "https://docs.google.com/spreadsheets/d/13AL_aWKj8e3dG4BGo6rIgdPpa8WGQzNwT-WL_pw8BUU/edit",
    "https://docs.google.com/spreadsheets/d/1pUxH_LejZYyrs5cSg27wStpxJkj_GJHUH-SluGwgBGI/edit",
    "https://docs.google.com/spreadsheets/d/1yqm0AjjYJVOMPtgYglroQkwjRH9tuXqgSg7iimcQC-w/edit",
    "https://docs.google.com/spreadsheets/d/1LfUPBI7FLVBpQ7nCoVwM__3E8QLBvGbf6A39B2Ng3sw/edit",
    "https://docs.google.com/spreadsheets/d/1bJMRAdHfco5K_PqHbBUQ2BRDLe0tkGmCsPHtkW6RIaA/edit",
    "https://docs.google.com/spreadsheets/d/1bsu9jA49lP386TXZD12RNqAxvj91UxQ87IdU8bMtiHs/edit"
]

# === GOOGLE API CLIENTS ===
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)
docs_service = build('docs', 'v1', credentials=creds)
sheets_service = build('sheets', 'v4', credentials=creds)

# === UTILS ===
def extract_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else url

def read_google_doc(doc_id):
    doc = docs_service.documents().get(documentId=doc_id).execute()
    content = []
    for el in doc.get('body', {}).get('content', []):
        if 'paragraph' in el:
            for elem in el['paragraph'].get('elements', []):
                content.append(elem.get('textRun', {}).get('content', ''))
    return ''.join(content).strip()

def read_google_sheet(sheet_id):
    sheet = gc.open_by_key(sheet_id)
    worksheet = sheet.sheet1
    data = worksheet.get_all_values()
    return "\n".join([", ".join(row) for row in data])

def get_all_documents_context():
    blocks = []
    for url in google_docs_urls:
        try:
            doc_id = extract_id(url)
            content = read_google_doc(doc_id)
            blocks.append(f"--- Document: {doc_id} ---\n{content}")
        except Exception as e:
            print(f"[DOC ERROR] {url}: {e}")
    for url in google_sheets_urls:
        try:
            sheet_id = extract_id(url)
            content = read_google_sheet(sheet_id)
            blocks.append(f"--- Sheet: {sheet_id} ---\n{content}")
        except Exception as e:
            print(f"[SHEET ERROR] {url}: {e}")
    return blocks

def split_text(text, max_length=500):
    sentences = text.split('. ')
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_length:
            current += sent + ". "
        else:
            chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def build_rag_index(blocks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_texts, all_sources = [], []

    for block in blocks:
        lines = block.split("\n")
        title = lines[0].replace("---", "").strip()
        content = "\n".join(lines[1:])
        chunks = split_text(content)
        all_texts.extend(chunks)
        all_sources.extend([title] * len(chunks))

    embeddings = model.encode(all_texts, convert_to_numpy=True)

    return embeddings, all_texts, all_sources

# === EXECUTION ===
if __name__ == "__main__":
    print("📥 Lecture des documents Google...")
    context_blocks = get_all_documents_context()

    print("🔍 Construction de l'index RAG...")
    embeddings, all_texts, all_sources = build_rag_index(context_blocks)

    print("💾 Sauvegarde des données...")
    with open("all_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open("all_texts.pkl", "wb") as f:
        pickle.dump(all_texts, f)

    with open("all_sources.pkl", "wb") as f:
        pickle.dump(all_sources, f)

    print("✅ Terminé. Données prêtes pour l'application.")
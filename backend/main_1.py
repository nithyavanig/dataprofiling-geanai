from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

class PDFRuleMiner:
    def __init__(self):
        self.raw_text = ""
        self.sentences = []
        self.rules = []
        self.entities = {}
        self.relations = []
        self.stop_words = set(stopwords.words('english'))
       
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
           
            self.raw_text = text
            self.sentences = sent_tokenize(text)
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {e}"
   
    def preprocess_text(self):
        """Clean and preprocess the extracted text"""
        if not self.raw_text:
            return "No text to process. Please extract text from PDF first."
       
        clean_sentences = []
        for sentence in self.sentences:
            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
            clean_sentence = re.sub(r'[^\w\s.,?!:;()\[\]{}"\'-]', '', clean_sentence)
            if len(clean_sentence) > 10:
                clean_sentences.append(clean_sentence)
       
        self.sentences = clean_sentences
        return f"Preprocessed text: retained {len(self.sentences)} meaningful sentences"
   
    def extract_entities_and_relationships(self):
        """Extract named entities and relationships using spaCy"""
        if not self.sentences:
            return "No sentences to process. Please preprocess text first."
       
        for sentence in self.sentences:
            doc = nlp(sentence)
           
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ not in self.entities:
                    self.entities[ent.label_] = []
                if ent.text not in self.entities[ent.label_]:
                    self.entities[ent.label_].append(ent.text)
           
            # Extract subject-verb-object triples as potential relationships
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                   
                    for subj in subjects:
                        subj_span = self._get_span(subj, doc)
                        for obj in objects:
                            obj_span = self._get_span(obj, doc)
                            relation = {
                                "subject": subj_span,
                                "predicate": token.text,
                                "object": obj_span,
                                "sentence": sentence
                            }
                            self.relations.append(relation)
       
        return f"Extracted {len(self.entities.keys())} entity types and {len(self.relations)} relationships"
   
    def _get_span(self, token, doc):
        """Get the full noun phrase for a given token"""
        if token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr"):
            for np in doc.noun_chunks:
                if token in np:
                    return np.text
        return token.text
   
    def mine_rules(self):
        """Extract rules based on the text patterns"""
        self.extract_entities_and_relationships()
       
        for relation in self.relations:
            if any(cond in relation["sentence"].lower() for cond in ["if", "when", "while", "unless", "until"]):
                parts = re.split(r'\b(if|when|while|unless|until)\b', relation["sentence"], flags=re.IGNORECASE)
                if len(parts) >= 3:
                    condition = parts[2].strip()
                    action = parts[0].strip() if parts[0].strip() else " ".join(parts[3:]).strip()
                    rule = {
                        "type": "conditional",
                        "condition": condition,
                        "action": action,
                        "original_sentence": relation["sentence"]
                    }
                    self.rules.append(rule)
           
            if " is " in relation["sentence"] or " are " in relation["sentence"] or " refers to " in relation["sentence"]:
                doc = nlp(relation["sentence"])
                for token in doc:
                    if token.lemma_ in ("be", "refer") and token.i > 0 and token.i < len(doc) - 1:
                        subject = doc[:token.i].text.strip()
                        definition = doc[token.i+1:].text.strip()
                        rule = {
                            "type": "definition",
                            "term": subject,
                            "definition": definition,
                            "original_sentence": relation["sentence"]
                        }
                        self.rules.append(rule)
           
            rule = {
                "type": "fact",
                "subject": relation["subject"],
                "predicate": relation["predicate"],
                "object": relation["object"],
                "original_sentence": relation["sentence"]
            }
            self.rules.append(rule)
       
        # Deduplicate rules
        unique_rules = []
        seen_sentences = set()
        for rule in self.rules:
            if rule["original_sentence"] not in seen_sentences:
                unique_rules.append(rule)
                seen_sentences.add(rule["original_sentence"])
       
        self.rules = unique_rules
        return f"Mined {len(self.rules)} unique rules from the text"
   
    def answer_question(self, question):
        """Find relevant rules to answer a given question"""
        if not self.rules:
            return "No rules available. Please mine rules first."
       
        question_doc = nlp(question)
       
        question_type = "unknown"
        question_focus = ""
       
        if question.lower().startswith("what is") or question.lower().startswith("what are"):
            question_type = "definition"
            question_focus = question[8:].strip().rstrip("?")
        elif question.lower().startswith("who is") or question.lower().startswith("who are"):
            question_type = "person"
            question_focus = question[7:].strip().rstrip("?")
        elif question.lower().startswith("how"):
            question_type = "process"
        elif question.lower().startswith("why"):
            question_type = "reason"
        elif question.lower().startswith("when"):
            question_type = "time"
        elif question.lower().startswith("where"):
            question_type = "location"
        elif "?" in question:
            for token in question_doc:
                if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ("nsubj", "dobj", "pobj"):
                    question_focus = token.text
                    break
       
        all_texts = [question] + [rule["original_sentence"] for rule in self.rules]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
       
        question_vector = tfidf_matrix[0:1]
        rule_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(question_vector, rule_vectors).flatten()
       
        top_indices = similarities.argsort()[-10:][::-1]
        relevant_rules = [self.rules[i] for i in top_indices if similarities[i] > 0.1]
       
        if question_type == "definition":
            definition_rules = [rule for rule in self.rules if rule["type"] == "definition" and
                               question_focus.lower() in rule.get("term", "").lower()]
            if definition_rules:
                return definition_rules[0]["original_sentence"]
       
        if relevant_rules:
            if len(relevant_rules) == 1:
                return relevant_rules[0]["original_sentence"]
            else:
                answer = "Based on the information I found:\n"
                for rule in relevant_rules:
                    answer += f"- {rule['original_sentence']}\n"
                return answer
        else:
            return "I don't have enough information to answer this question based on the provided document."


# API Endpoints

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the PDF Rule Miner API!"}), 200

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
       
        miner = PDFRuleMiner()
        text = miner.extract_text_from_pdf(file_path)
        return jsonify({"status": "success", "text": text}), 200
    return jsonify({"status": "error", "message": "Invalid file type."}), 400

@app.route('/preprocess', methods=['POST'])
def preprocess_text():
    miner = PDFRuleMiner()
    message = miner.preprocess_text()
    return jsonify({"status": "success", "message": message}), 200

@app.route('/mine_rules', methods=['POST'])
def mine_rules():
    miner = PDFRuleMiner()
    message = miner.mine_rules()
    return jsonify({"status": "success", "message": message}), 200

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get("question")
    if not question:
        return jsonify({"status": "error", "message": "Question is required."}), 400
   
    miner = PDFRuleMiner()
    answer = miner.answer_question(question)
    return jsonify({"status": "success", "answer": answer}), 200


if __name__ == "__main__":
    app.run(debug=True, port=8005)
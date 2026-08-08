import json
import re
import os

def clean_topic(section_title):
    # Remove leading numbers like "1.1 ", "3.2.1 ", "1.6.3.2 "
    topic = re.sub(r'^(\d+\.)+\d*\s*', '', section_title)
    # Remove leading bullets like "● " or " "
    topic = re.sub(r'^[●]\s*', '', topic)
    
    topic = topic.strip()
    
    # Lowercase if it's all caps
    if topic.isupper():
        topic = topic.lower()
    else:
        # Lowercase the first letter
        if topic:
            topic = topic[0].lower() + topic[1:]
            
    # Add appropriate article if missing
    if not topic.startswith(('le ', 'la ', 'les ', "l'", 'un ', 'une ', 'des ', 'de ')):
        if topic.startswith(('a', 'e', 'i', 'o', 'u', 'y', 'é', 'è', 'ê', 'à', 'â')):
            topic = "l'" + topic
        elif topic.endswith('s') and not topic.endswith('ss'):
            topic = "les " + topic
        else:
            topic = "le " + topic # Default to 'le', not perfect but better than nothing
            
    return topic

def generate_conversational_question(q_type, section_title):
    topic = clean_topic(section_title)
    
    templates = {
        "definition": f"Comment définit-on {topic} ?",
        "process": f"Quelle est la procédure concernant {topic} ?",
        "obligation": f"Quelles sont les obligations liées à {topic} ?",
        "deadline": f"Quels sont les délais ou moments clés pour {topic} ?",
        "amount": f"Quels sont les montants ou chiffres clés pour {topic} ?",
        "condition": f"Sous quelles conditions s'applique {topic} ?",
        "actor": f"Qui est responsable de {topic} ?",
        "contact": f"Qui doit-on contacter pour {topic} ?",
        "system": f"Comment fonctionne {topic} ?",
        "factual": f"Que faut-il savoir sur {topic} ?",
        "list": f"Quels sont les éléments inclus dans {topic} ?",
        "table": f"Quelles données trouve-t-on concernant {topic} ?",
        "section": f"Que contient la partie sur {topic} ?",
        "multi_chunk": f"Quelles sont les informations croisées sur {topic} ?"
    }
    
    return templates.get(q_type, f"Pouvez-vous m'en dire plus sur {topic} ?")

def process_file(input_file, output_file, limit=250):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    results = []
    for line in lines[:limit]:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            
            # Extract section title from section_path or question
            section_title = ""
            if "section_path" in obj and obj["section_path"]:
                section_title = obj["section_path"][-1]
            else:
                # Try to extract from the old robotic question
                match = re.search(r'«\s*(.*?)\s*»', obj.get("question", ""))
                if match:
                    section_title = match.group(1)
            
            if section_title:
                q_type = obj.get("question_type", "factual")
                new_question = generate_conversational_question(q_type, section_title)
                obj["question"] = new_question
                
            results.append(obj)
        except Exception as e:
            print(f"Error processing line: {e}")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            
    print(f"Processed {len(results)} lines for {output_file}")

files = [
    "golden_eval_drr_2025.jsonl",
    "golden_eval_drr_2026.jsonl",
    "golden_eval_drr_2027.jsonl"
]

base_dir = "data/eval/eval chunks 512 max"

for file in files:
    input_path = os.path.join(base_dir, file)
    output_path = os.path.join(base_dir, file.replace(".jsonl", "_refactored.jsonl"))
    process_file(input_path, output_path, limit=250)

from openai import OpenAI, RateLimitError, APIConnectionError
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import time
from collections import defaultdict
from flask_cors import CORS

request_log = defaultdict(list)
RATE_LIMIT = 10
RATE_PERIOD = 600  # 10 minutes

load_dotenv()
print("🔑 Loaded OpenAI key:", os.getenv("OPENAI_API_KEY")[:10], "..." if os.getenv("OPENAI_API_KEY") else "❌ NOT FOUND")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())
app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """
You are a professional visual prompt engineer.

You generate short, visually precise prompts for creating a photo-realistic background image. A product image will be composited into this background later using a masked overlay. Your task is to describe only the background setting.

You will be punished if you create the product yourself instead of leaving empty space so that the devloper can paste the product in the image. 

The product is not present in the generation process. Instead, the image must be constructed in such a way that a product can be placed into the center afterward, with seamless integration in terms of lighting, perspective, and realism.

Strict instructions:
- DO NOT describe or include the product.
- DO NOT place any object, furniture, person, or shadow in the center of the image.
- DO NOT include people, hands, marketing language, or storytelling.
- DO NOT include placeholder elements or central focal points.
- DO NOT mention the product name.

Focus entirely on:
- Creating a realistic environment (e.g. office, living room, physiotherapy space).
- Matching the selected camera angle (front, side, or close-up).
- Lighting and shadows that make the product blend naturally when added.
- Walls, floors, decor elements, and background texture.
- Ensuring the center of the image is visually clean and empty.
- The image must look like a square (2000x2000px) camera photograph.

The final image will be used as a background only. The product will be masked and added later. Your prompt defines the scene into which the product must seamlessly fit.
"""


@app.route("/generate-prompt", methods=["POST"])
def generate_prompt():
    ip = request.remote_addr
    now = time.time()
    # Filter requests within time window
    request_log[ip] = [t for t in request_log[ip] if now - t < RATE_PERIOD]
    if len(request_log[ip]) >= RATE_LIMIT:
        return jsonify({"error": "Rate limit exceeded. Please wait before trying again."}), 429
    request_log[ip].append(now)

    data = request.get_json()
    print("📦 Inkomende data:", data)
    print("Selected Template:", data.get("selectedTemplate"))  # Log the selected template
    template_instruction = ""
    if data.get("selectedTemplate") == "Bejaardenhuis":
        template_info = """
        Genereer een warme, uitnodigende woonkamer in een bejaardenhuis met lichtgrijze muren, houten vloer in licht eiken, en zachte daglichtinval van links. Ruimte vrij voor plaatsing van een product via compositioning. De ruimte moet volledig realistisch  ogen. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Sportschool":
        template_info = """
        Genereer een moderne, realistische fitnessruimte met rubberen vloer, matgrijze muren en zachte verlichting. Laat het interieur volledig afgewerkt ogen, maar zonder  mensen . Alleen sfeerbeeld, geen product. houdt ruimte vrij voor eentuele toevoeging van product via compositioning. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Kantoor":
        template_info = """
        Genereer een lichte, Scandinavische kantoorruimte met off-white muren, licht houten visgraatvloer en zacht daglicht via transparante gordijnen. Plaats een bureau zonder stoel, zodat wij later zelf via compositioning de stoel kunnen toevoegen, maar oop de afbeelding moet dus niet de stoel staan. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Fysiotherapie":
        template_info = """
        Genereer een cleane, professionele fysiotherapieruimte met behandelbanken aan de zijkant, neutrale wanden en anatomieposters. laat lege ruimte over voor de plaatsing van een product via compostioning. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Straat":
        template_info = """
        Genereer een rustige Nederlandse straat met realistische bestrating, groen en eventueel een lantaarnpaal aan de rand. Laat de compositie open, maar volledig realistisch zonder mensen of objecten die lijken op een product. Alleen sfeerbeeld. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Woonkamer":
        template_info = """
        Genereer een moderne woonkamer in Scandinavische stijl met houten vloer, lichte muren en warme belichting. De kamer mag volledig ingericht zijn. Zorg ervoor dat er egens op de voorgrond ruimte is zodat wij later via compositioning een product kunnen plakken in die ruimte. Geen producten genereren. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Hal":
        template_info = """
        Genereer een rustige, opgeruimde hal met lichtgrijze muren, houten vloer en natuurlijk licht. Laat geen centraal object of decoratie zien die als product geïnterpreteerd kan worden. Alleen visuele sfeer. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Badkamer":
        template_info = """
        Genereer een moderne inloopdouche met beige tegels en warm daglicht. Geen accessoires of zichtbare producten. De compositie moet een realistische setting tonen zonder producten. Resolutie: 2000 x 2000 pixels.
        """
    else:
        template_info = "Standaardafbeelding van het product in een neutrale, fotorealistische setting."


    user_prompt = f"""
Je taak is om een fotorealistische achtergrondafbeelding te genereren waarin later een product (de {data.get("productnaam", " ")}) geplakt zal worden door onze eigen designers. Dus niet zelf een product genereren. Gebruik het volgende camerastandpunt: {data.get("view", "")} zicht.

De achtergrond moet volledig realistisch en natuurlijk ogen, in lijn met de volgende omgeving:

{template_info}

⚠️ Belangrijke eisen:
- In het midden van de afbeelding plakken wij zelf een product, dus NIET ZELF het product genereren.
- De belichting en het perspectief moeten zo gekozen zijn dat het geplakte product later natuurlijk integreert.
- De afbeelding moet vierkant zijn: 2000 x 2000 pixels.

Verwijder de stoel of product die je hebt gegenereerd want die willen wij zelf later erin plakken

Verwijder het product maar laat de rest staan
"""

    print("📨 Gebruikersprompt:")
    print(user_prompt)
    if data.get("selectedTemplate") != "AI_generated":
        return jsonify({"prompt": user_prompt.strip()+ "\n \n Genereer dit beeld in 2000 X 2000 pixels"})
    try:
        user_prompt = f"""
Je taak is om een korte, visueel nauwkeurige prompt te genereren voor het maken van een fotorealistische achtergrondafbeelding. Deze afbeelding dient als sfeerbeeld waarin wij later een product ({data.get("productnaam", " ")}) handmatig zullen plaatsen via compositing. Jij mag het product zelf dus NIET genereren of beschrijven.

De achtergrond moet volledig realistisch, sfeervol en visueel consistent zijn met het volgende scenario:
{template_info}

⚠️ Belangrijke vereisten:
- Genereer GEEN objecten, meubels of elementen die op het product lijken.
- Geen mensen, handen of centrale objecten.
- Houd rekening met het gekozen camerastandpunt: {data.get("view", "")} zicht.
- Zorg dat belichting en perspectief zo zijn dat het geplakte product er natuurlijk in zal passen.
- De afbeelding moet vierkant zijn: 2000 x 2000 pixels.
- Dit is uitsluitend een achtergrond — het product voegen wij zelf toe.

Extra context of wensen:
{data.get("extraDescription", "")}
"""
        
        
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9,
            max_tokens=400
        )
        if not completion.choices or not completion.choices[0].message.content:
            print("❌ Geen geldige output ontvangen van GPT")
            return jsonify({"error": "OpenAI gaf geen geldige output"}), 500
        result = completion.choices[0].message.content.strip()
        return jsonify({"prompt": result})
    except RateLimitError as e:
        print("❌ Je hebt je OpenAI-quota overschreden:", e)
        return jsonify({"error": "Je OpenAI-tegoed is op. Controleer je abonnement of betaalinstellingen op platform.openai.com"}), 429
    except APIConnectionError as e:
        print("❌ Kan geen verbinding maken met OpenAI:", e)
        return jsonify({"error": "Verbindingsfout met OpenAI. Probeer het later opnieuw."}), 503
    except Exception as e:
        print("❌ Onverwachte backend error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Nieuwe route voor het genereren van een afbeelding via prompt + upload
@app.route("/generate-image", methods=["POST"])
def generate_image():
    prompt = request.form.get("prompt") + "now remove the product"

    if not prompt:
        return jsonify({"error": "Prompt is vereist"}), 400

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return jsonify({"image_url": response.data[0].url})
    except RateLimitError as e:
        print("❌ Je hebt je OpenAI-quota overschreden:", e)
        return jsonify({"error": "Je OpenAI-tegoed is op. Controleer je abonnement of betaalinstellingen op platform.openai.com"}), 429
    except APIConnectionError as e:
        print("❌ Kan geen verbinding maken met OpenAI:", e)
        return jsonify({"error": "Verbindingsfout met OpenAI. Probeer het later opnieuw."}), 503
    except Exception as e:
        print("❌ Onverwachte backend error bij /generate-image:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
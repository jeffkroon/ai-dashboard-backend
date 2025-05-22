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
        Toon een looprek in een ruime, veilige woonkameromgeving die is ingericht voor ouderen. De ruimte heeft warme houten vloeren, een lichtgrijze of beige muur, en is ingericht met comfortabele, klassieke meubels zoals een zachte fauteuil, een houten bijzettafel met een leeslamp, en een vloerkleed met antislip. Laat natuurlijk daglicht binnenvallen via een groot raam met transparante gordijnen. Het looprek staat iets links van het midden, volledig zichtbaar, met ongeveer 2/4 van de ruimte rechts leeg voor tekst of USP’s. De compositie is warm, huiselijk en functioneel, met kleine details zoals een plant op de vensterbank of een leesbril op tafel. Geen visuele afleiding zoals snoeren of rommel. Resolutie: 2000 x 2000 pixels. Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    elif data.get("selectedTemplate") == "Sportschool":
        template_info = """
        [kinesiotape] in een [sportschool], weergegeven in een ruime, realistische setting met natuurlijke belichting en een rustige, industriële of sportieve achtergrond. Zorg dat het product goed zichtbaar en scherp gepositioneerd is, met 2/4 lege ruimte aan de rechterzijde van het beeld voor het toevoegen van USP’s of tekst. De compositie moet licht en functioneel, uitnodigend maar doelgericht zijn, met eventueel subtiele decoratieve elementen zoals een moderne trainingsbank, rubber sportvloer of fitnessapparatuur op afstand — zolang ze het product niet visueel overstemmen. Vermijd visuele afleiding zoals drukke patronen of overmatig felle kleuren. De afbeelding moet een extra realistische look hebben, met natuurlijke schaduwen, levensechte materialen en fijne details. Afbeeldingsgrootte: 2000 x 2000 pixels.Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    elif data.get("selectedTemplate") == "Kantoor":
        template_info = """
        Toon een ruime, moderne kantoorhoek (voor deze bureaustoel)in Scandinavische stijl met een serene, opgeruimde uitstraling. De ruimte is afgewerkt met mat afgewerkte, off-white muren en een visgraatvloer van licht eikenhout. Links valt zacht natuurlijk daglicht binnen via een groot raam met transparante, vloerlange gordijnen. Rechts staat een minimalistisch houten bureau met een opengeklapte laptop, een zwart koffiekopje, een gesloten notitieboekje en een kleine kamerplant in een keramische pot. De achterwand bevat een zwart stalen glazen scheidingswand, die de werkruimte subtiel spiegelt en extra ruimtelijkheid toevoegt. Eén wandcontactdoos is zichtbaar op de muur.De compositie is hyperrealistisch, licht en professioneel, met voldoende open vloeroppervlak voor een stoel. Er is geen visuele afleiding zoals losse kabels, papier of persoonlijke rommel. Resolutie: 2000 x 2000 pixel. Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    elif data.get("selectedTemplate") == "Fysiotherapie":
        template_info = """
        Maak een afbeelding [rollator] in een [fysiopraktijk], weergegeven in een ruime, realistische setting met natuurlijke belichting en een rustige, neutrale achtergrond. Zorg dat het product goed zichtbaar en scherp gepositioneerd is, met 2/4 lege ruimte aan de rechterzijde van het beeld voor het toevoegen van USP’s of tekst. De compositie moet warm en licht, functioneel en uitnodigend zijn, met eventueel subtiele decoratieve elementen zoals een moderne behandelbank of neutraal meubilair op afstand — maar zonder planten of vloerkleden. De afbeelding moet een extra realistische look hebben, met natuurlijke schaduwen, levensechte materialen en fijne details. Afbeeldingsgrootte: 2000 x 2000 pixels.Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    elif data.get("selectedTemplate") == "Straat":
        template_info = """
        Genereer een rustige Nederlandse straat met realistische bestrating, groen en eventueel een lantaarnpaal aan de rand. Laat de compositie open, maar volledig realistisch zonder mensen of objecten die lijken op een product. Alleen sfeerbeeld. Resolutie: 2000 x 2000 pixels. Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    elif data.get("selectedTemplate") == "Woonkamer":
        template_info = """
        Place the product in a modern minimalist living room setting that's clean and devoid of clutter. The room should have neutral-toned walls with stylish yet subtle wall art. There should be a smooth wooden floor adding warmth to the room. In the background, include a contemporary sideboard with a few decor items like a vase or a sculpture, providing a homely yet stylish ambiance. The room should be filled with natural light coming from a large window on the right side of the product. The sunlight subtly illuminates the room, providing a soft and inviting atmosphere. The Dunimed Lichtgewicht Rollator zwart should remain in the center of the image, with the indoor lighting casting gentle shadows to accentuate its features without altering its original appearance.Remember to keep the image realistic and consistent, as if it was a real photo. Maintain space on the right side of the image for adding any additional text or USP's. The overall image format should be square (2000x2000px), with the product perfectly placed in the center.Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    elif data.get("selectedTemplate") == "Hal":
        template_info = """
         in een ruime, moderne (hal van een huis bij de kapstok / [alternatieve hal-indeling]). De hal heeft een realistische afmeting en is afgewerkt met (lichte muren en een houten vloer /) en een (rustige, opgeruimde / [alternatieve stijl,zoals Scandinavische]) indeling. Laat natuurlijk daglicht zacht invallen van (links / [alternatieve lichtinval]) om een warme, functionele sfeer te creëren. Plaats het product iets links van het midden op de vloer, volledig zichtbaar. Aan de rechterzijde blijft ongeveer 2/4 van de afbeelding leeg, bedoeld voor het toevoegen van USP’s of tekst. Zet een paar schoenen netjes ernaast om context te geven. De compositie is hyperrealistisch, warm en licht, zonder visuele afleiding zoals (trap, jassen of accessoires / [alternatieve elementen om uit te sluiten]). Resolutie: 2000 x 2000 pixels. Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    elif data.get("selectedTemplate") == "Badkamer":
        template_info = """
        Toon (douchekruk) in een ruime, moderne (inloopdouche. De ruimte is betegeld met warme, beige keramische wand- en vloertegels met een lichte zandtint. Laat natuurlijk daglicht zacht invallen van links om een uitnodigende, functionele sfeer te creëren. Plaats het product iets links van het midden, volledig zichtbaar. Laat aan de rechterzijde ongeveer 2/4 van de afbeelding leeg voor het toevoegen van tekst of USP’s. Zet één realistisch object zoals een bruine fles douchegel op de vloer. Voeg in het midden van de vloer een subtiel, verzonken doucheputje toe van geborsteld RVS met diepte, schaduwwerking en tegelverloop naar het afvoerputje. De compositie is hyperrealistisch, warm en licht, zonder visuele afleiding zoals planten of accessoires. Resolutie: 2000 x 2000 pixels. Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?
        """
    else:
        template_info = "Standaardafbeelding van het product in een neutrale, fotorealistische setting."


    user_prompt = f"""
{template_info}
"""

    print("📨 Gebruikersprompt:")
    print(user_prompt)
    if data.get("selectedTemplate") != "AI_generated":
        return jsonify({"prompt": user_prompt.strip()+ "\n \n Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?"})
    try:
        user_prompt = f"""
Je taak is om een korte, visueel nauwkeurige prompt te genereren voor het maken van een fotorealistische achtergrondafbeelding. Deze afbeelding dient als sfeerbeeld waarin wij later een product ({data.get("productnaam", " ")}) handmatig zullen plaatsen via compositing. Jij mag het product zelf dus NIET genereren of beschrijven.

De achtergrond moet volledig realistisch, sfeervol en visueel consistent zijn met het volgende scenario:
{template_info}

- Houd rekening met het gekozen camerastandpunt: {data.get("view", "")} zicht.
- Zorg dat belichting en perspectief zo zijn dat het geplakte product er natuurlijk in zal passen.
- De afbeelding moet vierkant zijn: 2000 x 2000 pixels.


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
# Stable Diffusion (mage2Image pipeline)
import inspect
import random
from typing import List, Optional, Union
import torch
from torch import autocast
from tqdm.auto import tqdm
import warnings
import pdb
from PIL import Image
from pathlib import Path
import intel_extension_for_pytorch as ipex

from diffusers import StableDiffusionImg2ImgPipeline

# DEFINE OPTIONS

genre_buzz_words = {
    "Drama": ["Emotional", "Intense", "Heartfelt", "Character-driven"],
    "Comedy": ["Hilarious", "Lighthearted", "Witty", "Feel-good"],
    "Romance": ["Passionate", "Love story", "Sweeping", "Chemistry", "Epic"],
    "Action": ["Explosive", "Adrenaline", "High-octane", "Heroic", "Thrilling"],
    "Thriller": ["Suspense", "Mystery", "Tension", "Intrigue", "Edge-of-your-seat"],
    "Sci-Fi": ["Futuristic", "Otherworldly", "Space adventure", "Technology"],
    "Crime": ["Criminal", "Investigation", "Noir", "Mystery", "Detective"],
    "Family": ["Wholesome", "All-ages", "playfull", "Charming", "Family-friendly"]
}

genre_visual_features = {
    "Drama": ["Emotional characters", "Subdued color palette", "Expressive lighting", "Symbolic imagery"],
    "Comedy": [ "Bright and cheerful colors", "Comic illustrations", "Visual gags"],
    "Romance": [ "Soft focus", "Warm color schemes", "Intimate moments", "Heart-shaped elements"],
    "Action": ["Explosions and action sequences", "Dynamic composition", "High-contrast visuals", "Heroic poses", "Epic scale"],
    "Thriller": ["Dark and moody lighting", "Mysterious atmosphere", "Intriguing silhouettes", "Foreboding shadows", "Tense expressions"],
    "Sci-Fi": ["Futuristic technology", "Space and otherworldly landscapes", "Sleek and modern designs", "Alien creatures", "Cyberpunk aesthetics"],
    "Crime": ["Noir style", "Detective with a magnifying glass", "Mysterious alleys", "Gritty urban settings", "Crime scene details"],
    "Family": ["Colorful and friendly setting", "Simple and playful designs", "All-ages appeal" ]
}

genre_character_appearance = {
    "Drama": [ "Natural and relatable appearance", "Subtle gestures", "Character depth"],
    "Comedy": ["Playful and humorous appearance", "Comic attire", "Silly or goofy expressions"],
    "Romance": ["Dreamy and in love expressions", "Beautiful and attractive appearance", "Sensual poses"],
    "Action": ["Determined and heroic expressions", "Confident and ready-for-action appearance","Worn-out or rugged looks", "Muscular and fit physique", "Dynamic action poses", "Weapons or tools"],
    "Thriller": ["Tense and anxious facial expressions","Worn-out or rugged looks", "Fearful or suspicious looks", "Mysterious appearance with hidden motives", "Close-ups of worried eyes", "Chasing or running scenes"],
    "Sci-Fi": ["Futuristic and non-human appearances", "Alien or robotic characters", "High-tech gadgets and attire", "Astonished or curious expressions", "Space and time travel elements"],
    "Crime": ["Detective-like appearances", "Serious and investigative expressions", "Worn-out or rugged looks", "Mysterious demeanor", "Clues and evidence in the background"],
    "Family": [ "Joyful and wholesome appearance", "Friendly and approachable looks colorful clothes", "All-ages appeal"]
}

decades_info = {
    "1930": {
        "Visual Features": ["Vintage sepia tone", "Art deco design", "Classic typography", "Elegance and glamour", "Ornate borders"],
        "Character Appearance": ["Sophisticated attire", "Elegant hairstyles", "Formal expressions", "Old Hollywood charm", "Elegance and class"],
        "Decade Buzz Words": ["Golden Age", "Nostalgia", "Classic Cinema", "Glamorous", "Vintage"]
    },
    "1950": {
        "Visual Features": ["Technicolor vibrancy", "Retro fonts", "Cinemascope widescreen", "Atomic era aesthetics", "Drive-in movie posters"],
        "Character Appearance": ["Iconic fashion", "Rockabilly style", "Smiles and charm", "Youthful exuberance", "Hollywood stars"],
        "Decade Buzz Words": ["Rock 'n' Roll", "Atomic Age", "Mid-Century Modern", "Innocence", "Rebellion"]
    },
    "1970": {
        "Visual Features": ["Psychedelic colors", "Funky fonts", "Groovy patterns", "Exploitation film style", "Grainy textures"],
        "Character Appearance": ["Bell-bottoms and fringe", "Afros and long hair", "Rebellious expressions", "Counterculture icons", "1970s cool"],
        "Decade Buzz Words": ["Disco Fever", "Peace and Love", "Cult Classics", "Outrageous", "Revolution"]
    },
    "1980": {
        "Visual Features": ["Neon lights", "Sci-Fi aesthetics", "Retro-futuristic designs", "Synthwave-inspired colors", "Pixel art"],
        "Character Appearance": ["80s fashion", "Big hair and bold makeup", "Action hero poses", "Reckless confidence", "Arcade vibes"],
        "Decade Buzz Words": ["Neon Noir", "Cyberpunk", "MTV Generation", "Radical", "Nostalgic"]
    },
    "1990": {
        "Visual Features": ["Colorful gradients", "Digital effects", "Blockbuster fonts", "90s tech imagery", "CD case style"],
        "Character Appearance": ["90s fashion trends", "Youthful and carefree", "Generation X attitudes", "Pop culture references", "Iconic sitcom smiles"],
        "Decade Buzz Words": ["Generation X", "Dot-com Bubble", "Y2K", "Fresh", "Nostalgia"]
    },
    "2000": {
        "Visual Features": ["Sleek and glossy designs", "Matrix-style green tint", "Metallic accents", "High-tech gadgets", "DVD cover aesthetics"],
        "Character Appearance": ["Y2K fashion", "Matrix-inspired attire", "Action hero expressions", "Tech-savvy look", "Cyberpunk vibes"],
        "Decade Buzz Words": ["Digital Revolution", "Y2K Panic", "New Millennium", "Futuristic", "Cutting-Edge"]
    },
    "2020": {
        "Visual Features": ["Minimalist design", "Flat and clean aesthetics", "Social media-inspired visuals", "Streaming platform style", "App-like icons"],
        "Character Appearance": ["Contemporary attire", "Diverse representation"],
        "Decade Buzz Words": ["Digital Age", "Virtual Reality",]
    },
    "2040": {
        "Visual Features": ["Advanced holographics", "Biotech aesthetics", "Cybernetic designs", "Immersive VR experiences", "Futuristic cityscapes"],
        "Character Appearance": ["Enhanced cybernetic features", "Augmented reality attire", "Expressive through AI", "Transhumanist ideals", "Futuristic beauty"],
        "Decade Buzz Words": ["Post-Human", "AI Dominance", "Tech Utopia", "Cybernetic Evolution", "Sustainable"]
    }
}



def stable_diff(image, genre, decade, title):
    # Using random.sample() to create different prompts everytime
    # Get the buzz words, visual features, and character appearance for Drama genre
    buzz_words = random.sample(genre_buzz_words[genre], 2)
    visual_features = random.sample(genre_visual_features[genre], 2)
    character_appearance = random.sample(genre_character_appearance[genre], 2)

    # Get the visual features, character appearance, and decade buzz words for the year 1950
    decade_visual_features = random.sample(decades_info[decade]["Visual Features"], 2)
    decade_character_appearance = random.sample(decades_info[decade]["Character Appearance"], 2)
    decade_buzz_words = random.sample(decades_info[decade]["Decade Buzz Words"], 2)

    # Create a prompt using the gathered information
    prompt = f"Remove text, Create a {genre} scene that embodies the {decade} era. Make it {', '.join(buzz_words)} with {', '.join(visual_features)} elements. same numbers of people from original picture with clear facial features , character appearance: {', '.join(character_appearance)}. Capture spirit of {', '.join(decade_buzz_words)}. {title}"
    negative_prompt = "(((text))), (((title))), (words), (letters), ((Characters)), ((numbers)),((unclear face)), ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, (((poorly drawn face))), ((unclear face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (((long neck))), ((fuzzy face)), (((hazy facial features)))"
    # LOAD PIPELINE
    device = "cuda"
    model_path = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=False
    )

    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(1024)

    with autocast("cuda"):
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, strength=.6, guidance_scale=7.5, generator=generator).images[0]

    return image

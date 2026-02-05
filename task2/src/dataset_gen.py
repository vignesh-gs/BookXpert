"""Generate train.jsonl (600) and eval.jsonl (60) deterministically. Eval disjoint from train."""
import json
import random
import re
from pathlib import Path
from typing import List, Tuple

from src.config import DATA_DIR, EVAL_JSONL, TRAIN_JSONL
from src.dataset_format import count_lines


GEN_SEED = 42

# Common ingredient combinations -> (recipe title, ingredients list, steps, time_min, tips)
RECIPE_TEMPLATES: List[Tuple[List[str], str, List[str], List[str], int, str]] = [
    (["egg", "onion"], "Masala Omelette",
     ["egg", "onion", "salt", "pepper", "oil", "green chilli"],
     ["Beat eggs with salt and pepper.", "Chop onion and chilli.", "Heat oil, sauté onion.", "Pour egg mix, cook both sides."],
     10, "Serve with toast."),
    (["egg", "tomato"], "Egg Tomato Curry",
     ["egg", "tomato", "onion", "garam masala", "oil", "salt"],
     ["Boil eggs, peel.", "Blend tomato and onion.", "Heat oil, add paste, cook.", "Add eggs, garam masala, salt."],
     25, "Garnish with coriander."),
    (["egg", "potato"], "Aloo Ande",
     ["egg", "potato", "onion", "turmeric", "cumin", "oil"],
     ["Boil potato, dice.", "Boil eggs, peel.", "Heat oil, fry cumin, onion.", "Add potato, spices, eggs, simmer."],
     35, "Serve with rice or roti."),
    (["rice", "vegetables"], "Vegetable Fried Rice",
     ["rice", "mixed vegetables", "soy sauce", "garlic", "oil", "salt"],
     ["Cook rice, cool.", "Stir-fry garlic in oil.", "Add vegetables, cook.", "Add rice, soy sauce, toss."],
     25, "Use day-old rice for best results."),
    (["rice", "chicken"], "Chicken Rice",
     ["rice", "chicken", "ginger", "garlic", "stock", "salt"],
     ["Marinate chicken with ginger, garlic.", "Cook rice in stock.", "Steam or bake chicken.", "Serve chicken on rice."],
     45, "Use bone-in chicken for flavour."),
    (["pasta", "tomato"], "Tomato Pasta",
     ["pasta", "tomato", "garlic", "basil", "olive oil", "salt"],
     ["Cook pasta al dente.", "Sauté garlic in oil.", "Add tomato, basil, salt.", "Toss with pasta."],
     20, "Reserve pasta water for sauce."),
    (["pasta", "chicken"], "Chicken Pasta",
     ["pasta", "chicken", "cream", "garlic", "parsley", "salt"],
     ["Cook pasta.", "Sauté chicken until golden.", "Add garlic, cream, simmer.", "Toss with pasta, parsley."],
     30, "Add Parmesan if desired."),
    (["chicken", "onion"], "Chicken Onion Curry",
     ["chicken", "onion", "yogurt", "spices", "oil", "salt"],
     ["Marinate chicken with yogurt and spices.", "Sauté onion until golden.", "Add chicken, cook.", "Simmer until tender."],
     40, "Serve with naan."),
    (["paneer", "tomato"], "Paneer Butter Masala",
     ["paneer", "tomato", "cream", "butter", "spices", "salt"],
     ["Blend tomato, cook with spices.", "Add butter, cream.", "Add paneer cubes, simmer."],
     30, "Garnish with cream and kasuri methi."),
    (["paneer", "onion"], "Kadai Paneer",
     ["paneer", "onion", "capsicum", "kadai masala", "oil", "salt"],
     ["Sauté onion, capsicum.", "Add kadai masala.", "Add paneer, toss."],
     25, "Serve with roti."),
    (["lentils", "rice"], "Dal Rice",
     ["lentils", "rice", "onion", "cumin", "turmeric", "salt"],
     ["Cook lentils with turmeric.", "Temper with cumin, onion.", "Cook rice separately.", "Serve dal with rice."],
     35, "Classic comfort meal."),
    (["lentils", "tomato"], "Tomato Dal",
     ["lentils", "tomato", "mustard", "curry leaves", "salt"],
     ["Cook lentils.", "Add tomato, salt.", "Temper with mustard, curry leaves."],
     30, "Serve with rice."),
    (["milk", "oats"], "Oats Porridge",
     ["milk", "oats", "honey", "nuts", "cinnamon"],
     ["Boil milk.", "Add oats, stir.", "Cook 5 min.", "Top with honey, nuts, cinnamon."],
     10, "Use rolled oats."),
    (["banana", "milk", "oats"], "Banana Oat Smoothie",
     ["banana", "milk", "oats", "honey", "ice"],
     ["Blend oats to powder.", "Add banana, milk, honey, ice.", "Blend until smooth."],
     5, "Best served cold."),
    (["banana", "flour"], "Banana Pancakes",
     ["banana", "flour", "egg", "milk", "baking powder", "salt"],
     ["Mash banana.", "Mix flour, egg, milk, baking powder, salt.", "Pour on hot pan, flip."],
     15, "Add chocolate chips optional."),
    (["flour", "egg"], "Simple Crepe",
     ["flour", "egg", "milk", "butter", "salt"],
     ["Mix flour, egg, milk, salt.", "Rest batter 10 min.", "Cook in buttered pan, thin layer.", "Flip once."],
     20, "Fill with fruit or cheese."),
    (["potato", "onion"], "Aloo Pyaz Sabzi",
     ["potato", "onion", "cumin", "turmeric", "coriander", "oil"],
     ["Dice potato, slice onion.", "Heat oil, cumin.", "Add onion, potato, spices.", "Cook until potato tender."],
     25, "Serve with roti."),
    (["potato", "tomato"], "Aloo Tamatar",
     ["potato", "tomato", "ginger", "garam masala", "oil", "salt"],
     ["Dice potato.", "Cook tomato, ginger to paste.", "Add potato, water, cook.", "Finish with garam masala."],
     30, "Garnish with coriander."),
    (["chicken", "tomato"], "Chicken Tomato Curry",
     ["chicken", "tomato", "onion", "ginger garlic paste", "spices", "oil"],
     ["Sauté onion, ginger garlic.", "Add tomato, spices.", "Add chicken, cook.", "Simmer until done."],
     45, "Serve with rice or roti."),
    (["vegetables", "rice"], "Vegetable Pulao",
     ["rice", "mixed vegetables", "cumin", "bay leaf", "ghee", "salt"],
     ["Sauté cumin, bay leaf in ghee.", "Add vegetables, rice.", "Add water, salt, cook."],
     35, "Use basmati for fragrance."),
    (["egg", "flour"], "Egg Paratha",
     ["egg", "flour", "onion", "green chilli", "salt", "oil"],
     ["Knead dough from flour.", "Mix egg, onion, chilli, salt.", "Stuff paratha, roll.", "Cook on tawa with oil."],
     25, "Serve with pickle."),
    (["milk", "banana"], "Banana Milk Shake",
     ["milk", "banana", "sugar", "ice"],
     ["Blend banana, milk, sugar.", "Add ice, blend.", "Serve cold."],
     5, "Add dates for sweetness."),
    (["oats", "banana"], "Banana Oatmeal",
     ["oats", "banana", "milk", "honey", "cinnamon"],
     ["Cook oats in milk.", "Slice banana on top.", "Drizzle honey, cinnamon."],
     10, "Add nuts for crunch."),
    (["paneer", "capsicum"], "Paneer Capsicum",
     ["paneer", "capsicum", "onion", "tomato", "spices", "oil"],
     ["Cube paneer, slice capsicum.", "Sauté onion, tomato, spices.", "Add paneer, capsicum, cook."],
     25, "Serve with naan."),
    (["lentils", "spinach"], "Palak Dal",
     ["lentils", "spinach", "onion", "garlic", "cumin", "salt"],
     ["Cook lentils.", "Blanch spinach, blend.", "Add to dal with tempering."],
     35, "Good with rice."),
    (["rice", "egg"], "Egg Fried Rice",
     ["rice", "egg", "soy sauce", "spring onion", "oil", "salt"],
     ["Cook rice, cool.", "Scramble egg, set aside.", "Stir-fry rice, add egg, soy sauce.", "Garnish with spring onion."],
     20, "Use cold rice."),
    (["potato", "flour"], "Aloo Paratha",
     ["potato", "flour", "cumin", "coriander", "salt", "ghee"],
     ["Boil potato, mash with spices.", "Knead dough.", "Stuff, roll, cook on tawa."],
     35, "Serve with curd."),
    (["tomato", "onion"], "Tomato Onion Curry",
     ["tomato", "onion", "mustard", "curry leaves", "turmeric", "salt"],
     ["Chop tomato, onion.", "Temper mustard, curry leaves.", "Add onion, tomato, turmeric, salt.", "Cook to thick gravy."],
     20, "Serve with rice."),
    (["chicken", "potato"], "Chicken Potato Curry",
     ["chicken", "potato", "onion", "tomato", "spices", "oil"],
     ["Brown chicken, set aside.", "Sauté onion, tomato, spices.", "Add potato, chicken, water.", "Simmer until tender."],
     50, "Serve with rice."),
    (["milk", "flour"], "Milk Cake / Basundi",
     ["milk", "sugar", "cardamom", "nuts"],
     ["Reduce milk on low heat.", "Add sugar, cardamom.", "Garnish with nuts."],
     45, "Stir constantly."),
    (["pasta", "vegetables"], "Vegetable Pasta",
     ["pasta", "mixed vegetables", "garlic", "olive oil", "parmesan", "salt"],
     ["Cook pasta.", "Sauté vegetables, garlic.", "Toss with pasta, parmesan."],
     25, "Add chilli flakes optional."),
    (["egg", "milk"], "Egg Custard", ["egg", "milk", "sugar", "vanilla"],
     ["Mix egg, milk, sugar, vanilla.", "Pour in dish.", "Bake in water bath."], 45, "Do not overbake."),
    (["flour", "tomato"], "Tomato Sauce Pizza Base", ["flour", "tomato", "yeast", "oil", "salt"],
     ["Make dough with flour, yeast, salt.", "Roll, add tomato, oil.", "Bake until crisp."], 35, "Preheat oven."),
    (["chicken", "flour"], "Fried Chicken", ["chicken", "flour", "spices", "oil"],
     ["Coat chicken with flour, spices.", "Heat oil.", "Fry until golden."], 30, "Double coat for crisp."),
    (["potato", "cheese"], "Cheesy Potato", ["potato", "cheese", "milk", "butter", "salt"],
     ["Boil potato, mash.", "Add cheese, milk, butter.", "Bake until golden."], 40, "Top with more cheese."),
    (["rice", "tomato"], "Tomato Rice", ["rice", "tomato", "onion", "spices", "oil"],
     ["Cook rice.", "Sauté onion, tomato, spices.", "Mix with rice."], 30, "Tamarind optional."),
    (["lentils", "onion"], "Onion Dal", ["lentils", "onion", "cumin", "garlic", "oil"],
     ["Cook lentils.", "Fry onion, cumin, garlic.", "Temper dal."], 35, "Serve with rice."),
    (["oats", "milk", "banana"], "Overnight Oats", ["oats", "milk", "banana", "honey"],
     ["Mix oats, milk, honey.", "Refrigerate overnight.", "Top with banana."], 5, "Add nuts."),
    (["egg", "rice", "vegetables"], "Vegetable Egg Rice", ["rice", "egg", "vegetables", "soy sauce", "oil"],
     ["Cook rice.", "Scramble egg.", "Stir-fry vegetables, add rice, egg, soy."], 25, "Use cold rice."),
    (["paneer", "rice"], "Paneer Pulao", ["paneer", "rice", "spices", "onion", "ghee"],
     ["Sauté paneer, set aside.", "Cook rice with spices, onion.", "Mix paneer in."], 40, "Basmati preferred."),
]

# Extra variations for diversity (title, ingredients, steps, time, tips)
MORE_RECIPES = [
    (["egg", "bread"], "Egg in a Hole", ["egg", "bread", "butter", "salt"],
     ["Cut hole in bread.", "Butter pan, toast bread.", "Crack egg in hole.", "Flip, cook."], 5, "Quick breakfast."),
    (["egg", "cheese"], "Cheese Omelette", ["egg", "cheese", "salt", "pepper", "butter"],
     ["Beat eggs.", "Melt butter, pour eggs.", "Add cheese, fold."], 8, "Use cheddar or paneer."),
    (["rice", "lentils"], "Khichdi", ["rice", "lentils", "cumin", "ghee", "turmeric", "salt"],
     ["Wash rice and lentils.", "Pressure cook with cumin, turmeric, salt.", "Temper with ghee."], 30, "Comfort food."),
    (["chicken", "rice"], "Chicken Biryani", ["chicken", "rice", "yogurt", "biryani masala", "onion", "saffron"],
     ["Marinate chicken.", "Par-boil rice.", "Layer rice and chicken.", "Dum cook."], 60, "Soak rice 30 min."),
    (["paneer", "spinach"], "Palak Paneer", ["paneer", "spinach", "cream", "garlic", "spices", "oil"],
     ["Blanch spinach, blend.", "Sauté garlic, add puree.", "Add paneer, cream."], 30, "Serve with naan."),
    (["banana", "egg"], "Banana Egg Pancake", ["banana", "egg", "cinnamon"],
     ["Mash banana, mix with egg, cinnamon.", "Pour on pan.", "Cook both sides."], 10, "No flour needed."),
    (["oats", "egg"], "Oats Egg Breakfast", ["oats", "egg", "milk", "salt"],
     ["Mix oats, egg, milk.", "Pour in pan.", "Cook like pancake."], 12, "High protein."),
    (["potato", "pasta"], "Potato Pasta", ["potato", "pasta", "garlic", "olive oil", "parsley"],
     ["Boil potato, dice. Cook pasta.", "Sauté garlic, add potato, pasta.", "Toss with parsley."], 25, "Rustic style."),
    (["tomato", "egg"], "Shakshuka", ["tomato", "egg", "onion", "paprika", "cumin", "salt"],
     ["Sauté onion, add tomato, spices.", "Simmer.", "Crack eggs into wells, cover."], 25, "Serve with bread."),
    (["flour", "milk"], "Pancakes", ["flour", "milk", "egg", "baking powder", "sugar", "butter"],
     ["Mix dry ingredients.", "Add milk, egg.", "Cook on buttered pan."], 15, "Stack and serve with syrup."),
]


# Conversational opener templates (assistant reply). Pick by hash(title) for reproducibility.
CONVERSATIONAL_OPENERS = [
    "You can make a **{}**! It's quick and easy.",
    "Great combo! I'd suggest **{}**.",
    "How about **{}**? Here's how to make it.",
    "One of my favorites with those ingredients is **{}**.",
    "You could try **{}**—it works really well.",
    "I'd go with **{}**. Here are the details.",
    "Perfect for a **{}**! Let me walk you through it.",
    "A **{}** would be lovely. Here you go.",
]


def _normalize_input(s: str) -> str:
    """Normalize for dedup: lowercase, collapse spaces, strip conversational phrases."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    phrases = [
        "i have ", "please suggest", "what can i cook with", "give me a recipe for", "suggest recipe for",
        "at home. what can i make?", "at home. what can i make", "what can i make?",
        "any recipe ideas?", "any recipe ideas", "what do you recommend?", "what do you recommend",
        "quick recipe with ", "suggest something with ", "for breakfast.", "for breakfast", "for dinner.",
        "for dinner", "got some ", "i only have ", "—what do you recommend", " any ideas?",
        "recipe with ", "cook with ", "make with ",
    ]
    for phrase in phrases:
        s = re.sub(re.escape(phrase), " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ingredients_to_input(ingredients: List[str], variant: int) -> str:
    """One of several input variants (comma list, conversational sentence, etc.)."""
    base = list(ingredients)
    random.shuffle(base)
    and_joined = " and ".join(base)
    comma_joined = ", ".join(base)
    # Mix of short and conversational variants
    variants = [
        lambda: ", ".join(base),
        lambda: " and ".join(base),
        lambda: " ".join(base),
        lambda: ", ".join(b.upper() if i % 2 == 0 else b for i, b in enumerate(base)),
        lambda: "I have " + ", ".join(base),
        lambda: "Please suggest a recipe with " + comma_joined,
        lambda: "What can I cook with " + and_joined,
        lambda: comma_joined + " and salt",
        lambda: "ingredients: " + comma_joined,
        lambda: " ".join(base[::-1]),
        lambda: comma_joined + ", pepper",
        lambda: "Recipe for " + and_joined,
        # Conversational full sentences
        lambda: "I have " + and_joined + " at home. What can I make?",
        lambda: "What can I cook with " + comma_joined + "?",
        lambda: "Got some " + comma_joined + ". Any recipe ideas?",
        lambda: "I only have " + and_joined + "—what do you recommend?",
        lambda: "Quick recipe with " + comma_joined + "?",
        lambda: "Suggest something with " + comma_joined + " for breakfast.",
        lambda: "I have " + comma_joined + " at home. Any ideas?",
        lambda: "What can I make with " + and_joined + "?",
        lambda: "Got " + comma_joined + ". What do you suggest?",
        lambda: "Need a recipe using " + comma_joined + ".",
        lambda: "Can you suggest a dish with " + and_joined + "?",
        lambda: "What's a good recipe with " + comma_joined + "?",
    ]
    return variants[variant % len(variants)]()


def _recipe_to_output(title: str, ingredients: List[str], steps: List[str], time_min: int, tips: str) -> str:
    """Opener + structured block (Recipe / Ingredients / Steps / Time / Tips)."""
    opener_idx = hash(title) % len(CONVERSATIONAL_OPENERS)
    opener = CONVERSATIONAL_OPENERS[opener_idx].format(title)
    structured = [
        f"Recipe: {title}",
        "Ingredients: " + ", ".join(ingredients),
        "Steps: " + " | ".join(steps),
        f"Time: {time_min} minutes",
        f"Tips: {tips}",
    ]
    return opener + "\n\n" + "\n".join(structured)


def _build_examples(seed: int) -> Tuple[List[dict], List[dict]]:
    """Build 600 train + 60 eval, disjoint."""
    random.seed(seed)
    templates = RECIPE_TEMPLATES + MORE_RECIPES
    all_records: List[dict] = []
    seen_inputs: set = set()

    # Many variants per recipe to get 660+ unique inputs
    for (ing_list, title, ings, steps, time_min, tips) in templates:
        for variant in range(16):
            inp = _ingredients_to_input(ing_list, variant)
            norm = _normalize_input(inp)
            if norm in seen_inputs:
                continue
            seen_inputs.add(norm)
            all_records.append({"input": inp, "output": _recipe_to_output(title, ings, steps, time_min, tips)})

    # Refill until we have at least 660 (with guard)
    for _ in range(2000):
        if len(all_records) >= 660:
            break
        ing_list, title, ings, steps, time_min, tips = random.choice(templates)
        inp = _ingredients_to_input(ing_list, random.randint(0, 99))
        norm = _normalize_input(inp)
        if norm not in seen_inputs:
            seen_inputs.add(norm)
            all_records.append({"input": inp, "output": _recipe_to_output(title, ings, steps, time_min, tips)})

    random.shuffle(all_records)
    eval_records = all_records[-60:]
    eval_norms = {_normalize_input(r["input"]) for r in eval_records}
    train_records = [r for r in all_records[:-60] if _normalize_input(r["input"]) not in eval_norms]
    # Refill train to 600 if needed (max 500 attempts)
    train_norms = {_normalize_input(r["input"]) for r in train_records}
    for _ in range(500):
        if len(train_records) >= 600:
            break
        ing_list, title, ings, steps, time_min, tips = random.choice(templates)
        inp = _ingredients_to_input(ing_list, random.randint(0, 99))
        norm = _normalize_input(inp)
        if norm not in eval_norms and norm not in train_norms:
            train_norms.add(norm)
            train_records.append({"input": inp, "output": _recipe_to_output(title, ings, steps, time_min, tips)})

    return train_records[:600], eval_records[:60]


def generate_dataset(
    train_path: Path = TRAIN_JSONL,
    eval_path: Path = EVAL_JSONL,
    seed: int = GEN_SEED,
) -> None:
    """Write train.jsonl and eval.jsonl; creates data dir if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_records, eval_records = _build_examples(seed)
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generate_dataset()
    print(f"Generated {count_lines(TRAIN_JSONL)} train, {count_lines(EVAL_JSONL)} eval")

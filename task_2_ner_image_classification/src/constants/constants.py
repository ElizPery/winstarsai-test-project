# Animals list for NER entity recognition training
NER_ANIMALS = [
    "Lion",
    "Elephant",
    "Giraffe",
    "Zebra",
    "Hippopotamus",
    "Rhinoceros",
    "Kangaroo",
    "Koala",
    "Giant Panda",
    "Wolf",
    "Dolphin",
    "Shark",
    "Whale",
    "Penguin",
    "Crocodile",
    "Chimpanzee",
    "Octopus",
    "Eagle",
    "Owl",
    "Snake"
]

# Templates for generating sentences with animal entities for NER training
NER_TEMPLATES = [
        "I saw a {} standing in the field.",
        "There is definitely a {} in the picture.",
        "Could this be a {}?",
        "Look at that friendly {}!",
        "The document mentions a wild {}.",
        "I think I heard a {} nearby.",
        "The subject of the photo is a {}.",
        "The quick {} jumped over the small fence.",
        "I think I prefer the smaller {} over the large one.",
        "What kind of {} is visible in the background?",
        "A massive, dark-colored {} was grazing near the river.",
        "Can you confirm that this image definitely features a {}?",
        "This is my pet {}, and it is very playful.",
        "They warned me about a rare {} that lives in the high mountains.",
        "I wonder if the wild animal in the photo is actually a {}.",
        "The conservation group tracked a young {} for several months.",
        "If I had to make an educated guess, I'd say the creature is a {}.",
        "The only sign of life was a frightened {} hiding in the bush.",
        "Have you ever seen a {} running at full speed?",
        "I am almost certain the mammal pictured is a {}.",
        "My grandmother used to own a farm with a very noisy {}.",
        "It looks like the brown {} is quietly eating grass.",
    ]

# Pre-trained model name for NER fine-tuning
MODEL_NAME = "bert-base-uncased"
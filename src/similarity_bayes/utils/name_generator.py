from itertools import product
from random import shuffle
from typing import Generator

_adjectives = [
    "angry",
    "anxious",
    "happy",
    "sad",
]

_animals = [
    "aardvark",
    "alpaca",
    "antelope",
    "armadillo",
    "beetle",
    "boar",
    "camel",
    "capybara",
    "cheetah",
    "cod",
    "crocodile",
    "crow",
    "dolphin",
    "dove",
    "dragonfly",
    "eel",
    "elk",
    "emu",
    "ferret",
    "flamingo",
    "gecko",
    "gerbil",
    "gnu",
    "hamster",
    "hedgehog",
    "hornet",
    "iguana",
    "impala",
    "jaguar",
    "kangaroo",
    "koala",
    "komodo",
    "lemur",
    "leopard",
    "llama",
    "lobster",
    "mallard",
    "manatee",
    "mink",
    "moongoose",
    "moth",
    "narwhal",
    "octopus",
    "ostrich",
    "otter",
    "owl",
    "parrot",
    "panda",
    "pelican",
    "penguin",
    "porcupine",
    "quail",
    "raccoon",
    "raven",
    "rhinoceros",
    "salamander",
    "seahorse",
    "sloth",
    "squid",
    "stingray",
    "toad",
    "toucan",
    "walrus",
    "wombat",
    "zebra",
]


_surnames = [
    "c. barbosa",
    "szwarcfiter",
    "maculan",
    "lattes",
    "chagas",
    "nicolelis",
    "f. pessoa",
    "gomide",
    "a. marques",
    "gonzalez",
    "coralina",
    "meireles",
    "barreto",
    "de assis",
    "hist",
    "gullar",
    "m. de jesus",
    "bilac",
    "suassuna",
    "donato",
    "ben jor",
    "duncan",
    "regina",
    "nascimento",
    "pascoal",
    "carrilho",
    "c. prestes",
    "cavaquinho",
    "buarque",
    "d. nazareth",
    "carvalho",
    "b. cubas",
    "linspector",
    "i. lara",
    "c. filho",
    "j. ferreira",
    "dos palmares",
    "veloso",
    "verissimo",
    "amado",
    "guerra-peixe",
    "villa-lobos",
    "v. filho",
    "gonzaga",
    "simonal",
    "nazareth",
    "sampaio",
    "vasconcelos",
    "jobim",
    "rabello",
    "gilberto",
    "de holanda",
    "brandao",
    "nunes",
    "costa",
]


def capitalize(word: str) -> str:
    if word == "de":
        return word

    if "-" in word:
        return "-".join([w.capitalize() for w in word.split("-")])

    return word.capitalize()


def total_names() -> int:
    return len(_animals) * len(_surnames)


def name_generator() -> Generator[str, None, None]:
    all_pairs = list(product(_animals, _surnames))
    shuffle(all_pairs)

    for animal, surname in all_pairs:
        words = (animal + " " + surname).split(" ")
        capitalized = [capitalize(w) for w in words]
        yield " ".join(capitalized)

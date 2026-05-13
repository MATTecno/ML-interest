"""
Gerador de perfis sintéticos para bootstrap do modelo.
Cria exemplos fictícios que claramente combinam ou não combinam com as preferências
do config.yaml. Serve apenas para o modelo ter dados iniciais antes do usuário
rotular perfis reais.
"""

import csv
import random
import yaml
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
SYNTHETIC_PATH = Path(__file__).parent.parent / "data" / "synthetic.csv"


NAMES_COMMON = ["Ana", "Julia", "Fernanda", "Mariana", "Beatriz", "Larissa",
                "Camila", "Leticia", "Gabriela", "Amanda"]

POSITIVE_BIOS = [
    "Apaixonada por viagens e aventuras ao ar livre. Adoro trilhas e novos horizontes.",
    "Amante de livros, café e boa conversa. Academia todo dia, sem falta.",
    "Adoro pets! Tenho dois cachorros. Gosto de cozinhar e assistir séries.",
    "Viajante de coração, já fui a 15 países. Amo fotografia e natureza.",
    "Leitora voraz, músico amadora e apaixonada por trilhas na montanha.",
    "Faço academia 5x por semana. Amo viajar e conhecer culturas novas.",
    "Tenho um gato chamado Biscuit. Trabalho com design e amo cinema.",
    "Criativa, espontânea e independente. Adoro cozinhar pratos novos.",
    "Fotógrafa nas horas vagas. Natureza e música são minha paz.",
    "Café de manhã, livro à tarde, série à noite. Academia é obrigação.",
]

NEGATIVE_BIOS = [
    "",
    "Procuro alguém sério para um relacionamento. Tenho filhos, eles são minha prioridade.",
    "Meu ex não me valorizou. Não aguento mais drama.",
    "Insta no bio. Só passo o @ pra quem for interessante.",
    "Snap no bio. Se quiser conversar vai no insta.",
    "Casamento é meu objetivo de vida.",
    "Já fui muito magoada. Preciso de alguém que me dê segurança.",
    "Mãe do Davi e da Clara. Eles são tudo pra mim.",
    "Nao uso esse app direito nao.",
    "Ex me ensinou muito. Agora só aceito o que mereço.",
]

POSITIVE_INTERESTS = ["academia", "viagem", "música", "livros", "pets",
                      "fotografia", "natureza", "culinária", "cinema"]
NEGATIVE_INTERESTS = ["festas", "barzinho", "agro", "sertanejo universitário"]


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_synthetic_profiles(n: int = 80) -> list[dict]:
    """Gera N perfis sintéticos balanceados (metade positivos, metade negativos)."""
    config = _load_config()
    age_min, age_max = config["preferences"]["age_range"]

    profiles = []
    half = n // 2

    # Perfis que devem ser curtidos (label=1)
    for _ in range(half):
        age = random.randint(age_min, age_max)
        interests = random.sample(POSITIVE_INTERESTS, k=random.randint(2, 5))
        profiles.append({
            "name": random.choice(NAMES_COMMON),
            "age": age,
            "bio": random.choice(POSITIVE_BIOS),
            "interests": ",".join(interests),
            "label": 1,
            "source": "synthetic",
        })

    # Perfis que NÃO devem ser curtidos (label=0)
    for _ in range(half):
        # Varia o motivo da rejeição
        reason = random.choice(["age", "bio", "interests", "mixed"])
        if reason == "age":
            candidates = []
            if age_min > 18:
                candidates.append(random.randint(18, age_min - 1))
            candidates.append(random.randint(age_max + 1, age_max + 8))
            age = random.choice(candidates)
        else:
            age = random.randint(age_min, age_max)

        if reason == "bio" or reason == "mixed":
            bio = random.choice(NEGATIVE_BIOS)
        else:
            bio = random.choice(POSITIVE_BIOS)

        if reason == "interests" or reason == "mixed":
            interests = random.sample(NEGATIVE_INTERESTS, k=random.randint(1, 3))
        else:
            interests = random.sample(POSITIVE_INTERESTS[:3], k=1)

        profiles.append({
            "name": random.choice(NAMES_COMMON),
            "age": age,
            "bio": bio,
            "interests": ",".join(interests),
            "label": 0,
            "source": "synthetic",
        })

    random.shuffle(profiles)
    return profiles


def save_synthetic(profiles: list[dict], path: Path = SYNTHETIC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "age", "bio", "interests", "label", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profiles)


def ensure_synthetic_exists(n: int = 80) -> Path:
    """Gera e salva dados sintéticos se ainda não existirem."""
    if not SYNTHETIC_PATH.exists():
        profiles = generate_synthetic_profiles(n)
        save_synthetic(profiles)
        print(f"[synthetic] {len(profiles)} perfis sintéticos gerados em {SYNTHETIC_PATH}")
    return SYNTHETIC_PATH

"""
Filtros absolutos que rejeitam perfis imediatamente, antes do modelo ML.
Se qualquer filtro disparar, o perfil é recusado sem análise adicional.

Configuráveis via config.yaml → hard_filters
"""

import re
import unicodedata
import yaml
from pathlib import Path
from logging_config import get_logger

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
logger = get_logger(__name__)

# ─── Nomes que claramente não são nomes reais em PT-BR ───────────────────────
_BUILTIN_FAKE_NAMES = {
    # Aparência física como nome
    "morena", "loira", "ruiva", "morena gata", "loira gata",
    "gordinha", "magrinha", "branquinha", "pretinha",
    # Elogios / apelativos
    "gata", "gatinha", "linda", "gostosa", "safada", "tesuda",
    "princesa", "deusa", "perfeita", "delicia", "delícia", "novinha",
    "musa", "diva",
    # Anonimato
    "sigilo", "anonima", "anônima", "anon", "sem nome", "incognita",
    "incógnita", "misteriosa", "misterio", "mistério", "segredo",
    "secreto", "secreta", "nao identificada", "não identificada",
    # Status civil/emocional como nome
    "solteira", "livre", "disponivel", "disponível",
    # Região/etnia como nome
    "nordestina", "paulistana", "carioca", "mineira", "baiana", "gaúcha",
    # Signos como nome
    "ariana", "taurina", "gemea", "gêmea", "canceriana", "leonina",
    "virginiana", "libriana", "escorpiana", "sagitariana",
    "capricorniana", "aquariana", "pisciana",
    # Palavras genéricas
    "modelo", "atriz", "influencer", "digital", "usuario", "usuária",
    "perfil", "conta", "pessoa",
}

# Palavras que, se contidas no nome, indicam fake
_BUILTIN_FAKE_NAME_WORDS = [
    "gata", "safada", "princesa", "deusa", "gostosa", "sigilo",
    "anon", "morena", "loira", "novinha",
]

# ─── Nomes masculinos brasileiros comuns ─────────────────────────────────────
# Checamos apenas o PRIMEIRO token do nome (ex: "João Silva" → "joao")
_MALE_NAMES = {
    # Clássicos PT-BR
    "joao", "jose", "carlos", "luis", "luiz", "pedro", "paulo", "marcos",
    "rafael", "bruno", "lucas", "thiago", "rodrigo", "felipe", "gabriel",
    "daniel", "guilherme", "mateus", "matheus", "henrique", "leandro",
    "anderson", "alexandre", "andre", "antonio", "arthur", "caio", "claudio",
    "cristiano", "david", "diogo", "douglas", "eduardo", "emerson",
    "fabricio", "fabio", "fernando", "flavio", "francisco", "geovane",
    "gilberto", "gustavo", "igor", "ivan", "jorge", "julio", "junior",
    "kevin", "leonardo", "luan", "marcelo", "marcio", "mario", "mauricio",
    "miguel", "murilo", "nelson", "nicolas", "patrick", "raimundo",
    "renato", "ricardo", "roberto", "romulo", "ronan", "ronaldo", "samuel",
    "sergio", "silvio", "tiago", "vagner", "vinicius", "vitor", "vithor",
    "wagner", "william", "willian", "welington", "wellington", "yuri",
    "kaique", "kaio", "kayke", "ryan", "alan", "allan", "alex",
    "diego", "erick", "erik", "everton", "ewerton", "ezequiel",
    "gerson", "giovani", "heitor", "hernan", "hudson", "hugo",
    "jean", "jefferson", "jhon", "joao", "jonatan", "jonathan",
    "jordan", "juliano", "kaua", "kelvin", "levi", "lincon", "lincoln",
    "lukas", "manoel", "manuel", "michael", "michel", "nathan",
    "olavo", "oscar", "otavio", "otto", "pablo", "rafa", "raul",
    "reginaldo", "reinaldo", "renan", "rene", "renzo", "rhuan", "ruan",
    "silas", "simao", "theo", "thales", "thalles", "tomas", "tony",
    "victor", "vini", "vin",
    # Apelidos masculinos comuns
    "gui"  # removido: gabi pode ser feminino — ok, nome completo
    "beto", "caco", "dado", "dede", "guto", "juca", "kiko",
    "lelo", "lico", "neto", "ninho", "nino", "pepe", "teco",
    "teto", "tito", "tuca", "xande", "zeca", "zico", "zico", "zito",
    # Nomes internacionais mais comuns no BR
    "adam", "christopher", "christian", "ethan", "henry", "jacob",
    "james", "jason", "john", "mark", "matthew", "michael", "noah",
    "peter", "robert", "thomas", "tyler", "zachary",
}

# ─── Indicadores trans ────────────────────────────────────────────────────────
_TRANS_KEYWORDS = [
    "mulher trans", "garota trans", "menina trans", "sou trans",
    "transgênero", "transgenero", "transexual", "travesti",
    "she/her", "ela/ela", "she/they", "ela/elas",
    # detecta "trans" isolado (com espaço, vírgula, ponto ou fim de linha ao redor)
    # para não bloquear palavras como "transparente" ou "transportar"
]

_TRANS_EMOJI = [
    "\U0001F3F3️‍⚧️",  # 🏳️‍⚧️ bandeira trans completa
    "⚧",    # ⚧ símbolo trans isolado
    "⚧️",
]

# Regex: "trans" como palavra inteira (não dentro de outra palavra)
# Aplicado sobre o texto já normalizado (_deep_norm), então sem acento
_TRANS_WORD_RE = re.compile(r"\btrans\b", re.IGNORECASE)

# Palavras que NÃO devem ser bloqueadas mesmo contendo "trans"
# (para evitar falsos positivos)
_TRANS_WHITELIST_RE = re.compile(
    r"\b(transparente|transporte|transformar|transmitir|tranquilo"
    r"|transacao|transicao|transitar|transferir|transplante"
    r"|transcender|transcricao|transbordar)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Lowercase + remove acentos."""
    t = text.lower().strip()
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


# Tabela de substituições de leetspeak / escrita ofuscada
_LEET_TABLE = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "@": "a", "$": "s", "!": "i", "+": "t", "8": "b",
    "q": "k",   # "qüeer" → "kueer"
})

def _deep_norm(text: str) -> str:
    """
    Normalização profunda para detectar variações intencionais de escrita:
    - Remove acentos (transgênero → transgenero)
    - Converte leetspeak (tr4ns → trans)
    - Remove separadores entre letras (t-r-a-n-s → trans, t.r.a.n.s → trans)
    - Colapsa caracteres repetidos (traaaans → trans)
    - Lowercase
    """
    t = text.lower()
    # Remove acentos
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    # Leetspeak
    t = t.translate(_LEET_TABLE)
    # Remove separadores NÃO-ESPAÇO entre letras (t-r-a-n-s → trans, t.r.a.n.s → trans)
    # Não remove espaços — isso quebraria palavras normais ("trans e feliz" → "transe feliz")
    t = re.sub(r"(?<=[a-z])[.\-_](?=[a-z])", "", t)
    # Colapsa letras repetidas (traaans → trans, safadaa → safada)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return t


def _load_config_filters() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f).get("hard_filters", {})
    except Exception:
        logger.exception("Falha ao carregar hard_filters do config")
        return {}


# ─── Detecção de nome falso ───────────────────────────────────────────────────

def is_fake_name(name: str) -> tuple[bool, str]:
    """
    Retorna (é_fake, motivo).
    Se é_fake=True, o perfil deve ser recusado imediatamente.
    """
    cfg = _load_config_filters()
    extra_fake_names = {_norm(n) for n in cfg.get("extra_fake_names", [])}
    extra_fake_words = [_norm(w) for w in cfg.get("extra_fake_name_words", [])]

    name_stripped = name.strip()
    name_norm = _norm(name_stripped)

    # Nome vazio
    if not name_norm:
        return True, "Nome vazio"

    # Nome muito curto — menos de 3 caracteres (bloqueia "CH", "KR", "X", etc.)
    if len(name_norm) < 3:
        return True, f"Nome muito curto ('{name_stripped}')"

    # Contém dígito (ex: "Carol123", "Gata69")
    if re.search(r"\d", name_stripped):
        return True, f"Nome com número: '{name_stripped}'"

    # Contém emoji
    emoji_pattern = re.compile(
        "[\U00010000-\U0010FFFF]|[\U0001F300-\U0001F9FF]", flags=re.UNICODE
    )
    if emoji_pattern.search(name_stripped):
        return True, f"Nome com emoji: '{name_stripped}'"

    # Tudo maiúsculo com 2+ chars (ex: "CH", "MORENA GATA", "KR")
    only_letters = name_stripped.replace(" ", "")
    if only_letters == only_letters.upper() and only_letters.isalpha() and len(only_letters) >= 2:
        return True, f"Nome em maiúsculo: '{name_stripped}'"

    # Na lista built-in de nomes falsos (usando normalização profunda)
    name_deep = _deep_norm(name_stripped)
    all_fake_names = _BUILTIN_FAKE_NAMES | extra_fake_names
    if name_norm in all_fake_names or name_deep in all_fake_names:
        return True, f"Nome falso na lista: '{name_stripped}'"

    # Contém palavras indicativas de fake (também com deep norm)
    all_fake_words = _BUILTIN_FAKE_NAME_WORDS + extra_fake_words
    for word in all_fake_words:
        if word in name_norm.split() or word in name_deep.split():
            return True, f"Nome contém palavra fake '{word}': '{name_stripped}'"

    return False, ""


# ─── Detecção de nome masculino ──────────────────────────────────────────────

def is_male_name(name: str) -> tuple[bool, str]:
    """
    Verifica se o primeiro token do nome é um nome masculino.
    Retorna (é_masculino, motivo).
    """
    cfg = _load_config_filters()
    extra_male = {_norm(n) for n in cfg.get("extra_male_names", [])}

    first_token = _norm(name.strip().split()[0]) if name.strip() else ""
    if not first_token:
        return False, ""

    all_male = _MALE_NAMES | extra_male
    if first_token in all_male:
        return True, f"Nome masculino: '{name.strip()}'"

    return False, ""


# ─── Detecção trans ───────────────────────────────────────────────────────────

def has_trans_indicator(
    bio: str,
    name: str = "",
    descriptors: dict | None = None,
) -> tuple[bool, str]:
    """
    Retorna (tem_indicador, motivo).
    Verifica bio, nome e descritores estruturados do Tinder.
    """
    combined = f"{bio} {name}"
    combined_deep = _deep_norm(combined)

    # Emojis trans
    for emoji in _TRANS_EMOJI:
        if emoji in bio or emoji in name:
            return True, "Emoji de bandeira/símbolo trans detectado"

    # Palavras compostas — checadas no texto normalizado (sem acento)
    combined_norm = _norm(combined)
    for kw in _TRANS_KEYWORDS:
        kw_norm = _norm(kw)
        if kw_norm in combined_norm or kw_norm in combined_deep:
            return True, f"Indicador trans: '{kw}'"

    # "trans" como palavra isolada — também no deep_norm (captura t-r-a-n-s, tr4ns, etc.)
    for text_to_check in (combined, combined_deep):
        if _TRANS_WORD_RE.search(text_to_check):
            # Verifica se não é falso positivo (transparente, transporte, etc.)
            clean = _TRANS_WHITELIST_RE.sub("", text_to_check)
            if _TRANS_WORD_RE.search(clean):
                return True, "Palavra 'trans' isolada na bio/nome"

    # Descritores estruturados do Tinder (campo "Identidade de gênero")
    if descriptors:
        for key, val in descriptors.items():
            if "gênero" in key.lower() or "genero" in key.lower() or "identidade" in key.lower():
                val_lower = val.lower()
                if any(t in val_lower for t in ["trans", "não binário", "nao binario"]):
                    return True, f"Descritor de gênero: '{key}: {val}'"

    return False, ""


# ─── Filtros absolutos baseados na foto ──────────────────────────────────────

def apply_photo_filters(photo_features: dict) -> tuple[bool, str]:
    """
    Aplica filtros absolutos baseados nas features extraídas da foto.
    Retorna (deve_recusar, motivo).

    Chamado APÓS analyze_photo(), antes do modelo ML.
    """
    if not photo_features:
        return False, ""

    # Falha de download/CDN/DeepFace é "foto indisponível", não "sem rosto".
    # Nesses casos deixamos o ML decidir com features neutras em vez de passar
    # automaticamente um perfil que talvez seja válido.
    if photo_features.get("_analysis_failed"):
        logger.warning(
            "Filtro de foto ignorado por analise indisponivel: reason=%s failed=%s timed_out=%s",
            photo_features.get("_failure_reason", ""),
            photo_features.get("_photos_failed", 0),
            photo_features.get("_photos_timed_out", 0),
        )
        return False, ""

    cfg = _load_config_filters()
    try:
        min_brightness = float(cfg.get("photo_min_brightness", 0) or 0)
    except Exception:
        min_brightness = 0.0
    if min_brightness > 0 and "photo_image_brightness" in photo_features:
        try:
            brightness = float(photo_features.get("photo_image_brightness", 0.5) or 0.5)
            contrast = float(photo_features.get("photo_image_contrast", 0.5) or 0.5)
        except Exception:
            brightness = 0.5
            contrast = 0.5
        if brightness < min_brightness and contrast <= 0.18:
            return (
                True,
                f"Foto muito escura/preta ({brightness * 100:.0f}% brilho < {min_brightness * 100:.0f}%)",
            )

    # Nenhum rosto detectado — foto de paisagem, texto, ou qualidade muito baixa
    if not photo_features.get("photo_has_face", 1):
        return True, "Nenhum rosto detectado na foto"

    try:
        min_woman_conf = float(cfg.get("photo_min_woman_confidence", 0) or 0)
    except Exception:
        min_woman_conf = 0.0
    if min_woman_conf > 0 and "photo_woman_confidence" in photo_features:
        try:
            woman_conf = float(photo_features.get("photo_woman_confidence", 0.5) or 0.5)
        except Exception:
            woman_conf = 0.5
        if woman_conf < min_woman_conf:
            return (
                True,
                f"Confiança visual feminina baixa ({woman_conf * 100:.0f}% < {min_woman_conf * 100:.0f}%)",
            )

    return False, ""


# ─── Ponto de entrada principal ───────────────────────────────────────────────

def apply_hard_filters(profile: dict) -> tuple[bool, str]:
    """
    Aplica todos os filtros absolutos ao perfil.
    Retorna (deve_recusar, motivo).

    profile deve ter: name, bio, interests (list), _descriptors (dict)
    """
    name = profile.get("name", "") or ""
    bio = profile.get("bio", "") or ""
    descriptors = profile.get("_descriptors") or {}

    fake, reason = is_fake_name(name)
    if fake:
        logger.info("Filtro hard aplicado: name=%r reason=%s", name, reason)
        return True, f"Nome inválido — {reason}"

    male, reason = is_male_name(name)
    if male:
        logger.info("Filtro hard aplicado: name=%r reason=%s", name, reason)
        return True, f"Nome masculino — {reason}"

    trans, reason = has_trans_indicator(bio, name, descriptors)
    if trans:
        logger.info("Filtro hard aplicado: name=%r reason=%s", name, reason)
        return True, f"Filtro trans — {reason}"

    return False, ""

"""
Interface de linha de comando principal do Tinder-IA.
Execute: python src/cli.py
"""

import sys
import yaml
from pathlib import Path

# Adiciona src/ ao path para imports relativos funcionarem
sys.path.insert(0, str(Path(__file__).parent))

import model as mdl
from logging_config import get_logger, setup_logging
from synthetic import ensure_synthetic_exists
from explainer import explain, format_stats


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
logger = get_logger(__name__)


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        logger.exception("Falha ao carregar config no CLI")
        raise


def _input_profile() -> dict:
    """Coleta dados de um perfil do Tinder via terminal."""
    print()
    print("  Preencha os dados do perfil:")
    print("  (Deixe em branco se não tiver a informação)")
    print()

    name = input("  Nome        : ").strip()
    age_str = input("  Idade       : ").strip()
    bio = input("  Bio         : ").strip()
    interests_raw = input("  Interesses  : (separe por vírgula) ").strip()

    try:
        age = int(age_str)
    except ValueError:
        age = 0

    interests = [i.strip() for i in interests_raw.split(",") if i.strip()] if interests_raw else []

    return {"name": name, "age": age, "bio": bio, "interests": interests}


def _ensure_model() -> dict:
    """Garante que o modelo existe e está treinado. Retorna model_data."""
    model_data = mdl.load_model()
    if model_data is None:
        print("\n  [modelo] Treinando modelo pela primeira vez...")
        ensure_synthetic_exists()
        df = mdl.load_all_data()
        mdl.train_model(df)
        model_data = mdl.load_model()
        print(f"  [modelo] Modelo treinado com {model_data['n_samples']} exemplos ({model_data['model_type']})")
    return model_data


def mode_predict():
    """Modo 1: O modelo decide se deve curtir ou não."""
    print("\n  ── Modo: O modelo decide ──")
    profile = _input_profile()
    model_data = _ensure_model()

    result = mdl.predict(profile, model_data)
    print()
    print(explain(result))

    # Pergunta se a decisão foi correta → adiciona ao dataset real
    print("  A decisão foi correta? (s/n/pular)")
    answer = input("  > ").strip().lower()
    if answer in ("s", "n"):
        label = 1 if answer == "s" else (0 if result["decision"] == "CURTIR" else 1)
        config = _load_config()
        retrain_every = config["model"]["retrain_every"]
        count_before = mdl.count_trainable_real_profiles()
        count_after = mdl.save_labeled_profile(profile, label)
        count_after_trainable = mdl.count_trainable_real_profiles()
        print(f"  Salvo! ({count_after} perfis reais no dataset)")

        if mdl.should_retrain(count_before, count_after_trainable, retrain_every):
            print("  Retreinando modelo com os novos dados...")
            mdl.train_model()
            new_data = mdl.load_model()
            print(f"  Modelo atualizado: {new_data['model_type']} | {new_data['n_samples']} exemplos")


def mode_label():
    """Modo 2: Usuário rotula um perfil (você decide, algoritmo aprende)."""
    print("\n  ── Modo: Rotular perfil ──")
    print("  Você avalia o perfil e diz se curtiu ou não.")
    print("  O modelo aprende com suas escolhas.\n")

    profile = _input_profile()

    print()
    print("  Você curtiu esse perfil? (s/n)")
    answer = input("  > ").strip().lower()
    while answer not in ("s", "n"):
        print("  Digite 's' para curtiu ou 'n' para não curtiu.")
        answer = input("  > ").strip().lower()

    label = 1 if answer == "s" else 0
    config = _load_config()
    retrain_every = config["model"]["retrain_every"]
    count_before = mdl.count_trainable_real_profiles()
    count_after = mdl.save_labeled_profile(profile, label)
    count_after_trainable = mdl.count_trainable_real_profiles()

    print(f"\n  Salvo! ({count_after} perfis reais no dataset)")

    if mdl.should_retrain(count_before, count_after_trainable, retrain_every):
        print("  Retreinando modelo com os novos dados...")
        mdl.train_model()
        new_data = mdl.load_model()
        print(f"  Modelo atualizado: {new_data['model_type']} | {new_data['n_samples']} exemplos")
    else:
        remaining = retrain_every - (count_after_trainable % retrain_every)
        print(f"  (Próximo retreino em {remaining} perfil(is))")


def mode_stats():
    """Modo 3: Estatísticas do projeto."""
    n_real = mdl.count_real_profiles()
    synthetic_path = Path(__file__).parent.parent / "data" / "synthetic.csv"
    n_synthetic = len(mdl._load_csv(synthetic_path)) if synthetic_path.exists() else 0
    model_data = mdl.load_model()
    model_type = model_data["model_type"] if model_data else "não treinado"
    print(format_stats(n_real, n_synthetic, model_type))


def mode_evaluate():
    """Modo 4: avaliação offline e calibração operacional."""
    from model_evaluation import evaluate_profiles, format_markdown_report, LATEST_JSON, LATEST_MD

    print("\n  ── Modo: Avaliar modelo ──")
    report = evaluate_profiles(config=_load_config(), write_reports=True)
    print()
    print(format_markdown_report(report))
    print(f"  Relatórios salvos em:\n  - {LATEST_JSON}\n  - {LATEST_MD}")


def main():
    setup_logging()
    logger.info("CLI iniciado")
    print()
    print("  ████████╗██╗███╗   ██╗██████╗ ███████╗██████╗      ██╗ █████╗ ")
    print("     ██╔══╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗    ██╔╝██╔══██╗")
    print("     ██║   ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝   ██╔╝ ███████║")
    print("     ██║   ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗  ██╔╝  ██╔══██║")
    print("     ██║   ██║██║ ╚████║██████╔╝███████╗██║  ██║ ██╔╝   ██║  ██║")
    print("     ╚═╝   ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝    ╚═╝  ╚═╝")
    print("                         I A  -  classificador de perfis")
    print()

    # Garante dados sintéticos na primeira execução
    ensure_synthetic_exists()

    while True:
        n_real = mdl.count_real_profiles()
        model_data = mdl.load_model()

        print()
        print("  ─────────────────────────────────────────────")
        if model_data:
            print(f"  Modelo: {model_data['model_type']} | {model_data['n_samples']} exemplos | {n_real} reais")
        else:
            print("  Modelo: ainda não treinado")
        print()
        print("  1. Deixar o modelo decidir (curtir ou não)")
        print("  2. Rotular um perfil (você decide, modelo aprende)")
        print("  3. Ver estatísticas")
        print("  4. Avaliar modelo/calibração")
        print("  0. Sair")
        print()

        choice = input("  Escolha: ").strip()

        if choice == "1":
            logger.info("CLI modo predict selecionado")
            mode_predict()
        elif choice == "2":
            logger.info("CLI modo label selecionado")
            mode_label()
        elif choice == "3":
            logger.info("CLI modo stats selecionado")
            mode_stats()
        elif choice == "4":
            logger.info("CLI modo evaluate selecionado")
            mode_evaluate()
        elif choice == "0":
            logger.info("CLI encerrado pelo usuario")
            print("\n  Até mais!\n")
            break
        else:
            print("\n  Opção inválida. Digite 1, 2, 3 ou 0.")


if __name__ == "__main__":
    main()

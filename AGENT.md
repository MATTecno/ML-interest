# AGENT.md — Tinder-IA

Documentação para IAs que trabalham neste projeto. Leia antes de sugerir mudanças.

---

## Regra obrigatória: ler todos os contextos de IA do projeto ao iniciar

**Independente do prompt recebido**, ao iniciar qualquer conversa verifique se os arquivos abaixo existem e leia cada um que encontrar. Eles podem conter instruções complementares, decisões de design ou restrições importantes.

| Arquivo | Ferramenta |
|---|---|
| `AGENT.md` | OpenAI Codex, Cursor Agent, geral |
| `AGENTS.md` | OpenAI Codex (alternativo) |
| `CLAUDE.md` | Claude Code (Anthropic) |
| `GEMINI.md` | Gemini CLI (Google) |
| `.cursorrules` | Cursor (legado) |
| `.cursor/rules/*.mdc` | Cursor (atual) |
| `.github/copilot-instructions.md` | GitHub Copilot |
| `.windsurfrules` | Windsurf (Codeium) |
| `.aider.conf.yml` / `CONVENTIONS.md` | Aider |

Se algum desses arquivos existir e você ainda não tiver lido, leia antes de prosseguir.

---

## Regra obrigatória: verificar alertas ao iniciar qualquer conversa

**Independente do prompt recebido**, sempre leia o conteúdo de `data/alerts/` antes de qualquer outra ação.

- Se houver arquivos `.md` na pasta, relate cada um ao usuário: data/hora, severidade, e o erro reportado.
- Só então prossiga com a tarefa solicitada.
- Se a pasta estiver vazia ou não existir, não mencione nada e siga normalmente.
- O sistema também grava alertas `Resource high usage` quando CPU ≥ `swiper.cpu_alert_threshold_percent` (padrão 80%) ou memória ≥ `swiper.mem_alert_threshold_percent` (padrão 85%). O `log_monitor.py` detecta esse padrão automaticamente.

```bash
ls data/alerts/
```

---

## O que é este projeto

Classificador de perfis do Tinder baseado em Machine Learning. Dado um perfil (nome, bio, idade, interesses), o sistema decide se o usuário deve curtir ou não, e explica o motivo.

**Objetivo principal:** projeto de estudo de ML. O usuário é iniciante em IA e quer aprender o pipeline completo: coleta de dados → features → treino → predição → explicação.

---

## Stack

- **Python 3.11+**
- **scikit-learn** — Pipeline / LogisticRegression / validação
- **LightGBM** — classificador principal quando há dados suficientes
- **pandas / numpy** — manipulação de dados
- **TextBlob** — análise de sentimento da bio
- **PyYAML** — configuração de preferências
- **tabulate** — formatação de tabelas no terminal

---

## Estrutura de arquivos

```
tinder-IA/
├── AGENT.md               ← este arquivo
├── requirements.txt       ← dependências Python
├── config.yaml            ← preferências do usuário (editável sem código)
├── data/
│   ├── profiles.csv       ← perfis reais rotulados pelo usuário (cresce com o uso)
│   └── synthetic.csv      ← perfis sintéticos gerados na primeira execução
├── src/
│   ├── features.py        ← engenharia de features (núcleo do sistema)
│   ├── model.py           ← treino, predição, serialização
│   ├── synthetic.py       ← gerador de dados sintéticos para bootstrap
│   ├── explainer.py       ← formatação de feature importance
│   ├── cli.py             ← ponto de entrada interativo (python src/cli.py)
│   └── import_json.py     ← importa JSON copiado da aba Network do browser
└── models/
    └── classifier.pkl     ← modelo treinado serializado (gerado em runtime)
```

---

## Como rodar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. (Apenas na primeira vez) Baixar corpus do TextBlob
python3 -m textblob.download_corpora

# 3. Modo interativo (entrada manual perfil a perfil)
python3 src/cli.py

# 4. Modo batch — processa JSON copiado da aba Network do browser
python3 src/import_json.py tinder_response.json
python3 src/import_json.py tinder_response.json --resumido  # sem explicação detalhada
```

### Como obter o JSON do Tinder (sem usar API diretamente)

1. Abra o Tinder no browser (Chrome/Firefox)
2. F12 → aba **Network**
3. Navegue pelos perfis até aparecer uma requisição com `recs` ou `v2/recs` na URL
4. Clique na requisição → aba **Response**
5. Selecione tudo (Ctrl+A) e copie
6. Cole em um arquivo `tinder_response.json`
7. Execute `python3 src/import_json.py tinder_response.json`

`data/synthetic.csv` está mantido apenas com cabeçalho por decisão do usuário.
Não regenere os 80 perfis sintéticos salvo pedido explícito; `ensure_synthetic_exists()`
só recria se o arquivo for removido.

---

## Fluxo de uso

```
Usuário abre o CLI
  ├─ Opção 1: Modelo decide
  │     ├─ Usuário insere perfil (nome, idade, bio, interesses)
  │     ├─ Modelo retorna CURTIR / NÃO CURTIR + confiança + explicação
  │     └─ Usuário confirma se acertou → salva no dataset real
  ├─ Opção 2: Usuário rotula
  │     ├─ Usuário insere perfil + diz se curtiu ou não
  │     └─ Sistema salva e retreina quando necessário
  └─ Opção 3: Estatísticas
```

---

## Pipeline de ML

### 1. Features extraídas de cada perfil

| Feature | Tipo | Descrição |
|---|---|---|
| `age_in_range` | binário | Idade dentro do intervalo de preferência |
| `age_distance` | contínuo | Distância normalizada do centro da faixa |
| `bio_length` | contínuo | Tamanho da bio em caracteres |
| `bio_word_count` | contínuo | Número de palavras na bio |
| `bio_has_min_length` | binário | Bio tem tamanho mínimo definido no config |
| `bio_positive_kw` | contínuo | Contagem de palavras positivas na bio |
| `bio_negative_kw` | contínuo | Contagem de palavras negativas na bio |
| `bio_sentiment` | contínuo (-1 a 1) | Sentimento geral da bio (TextBlob) |
| `interests_count` | contínuo | Total de interesses listados |
| `interests_overlap` | contínuo | Interesses em comum com as preferências |
| `name_length` | contínuo | Tamanho do nome |
| `name_in_disliked` | binário | Nome na lista negativa do config |
| `photo_face_smile_score` | contínuo (0 a 1) | Leitura leve de sorriso/expressão via DeepFace emotion |
| `photo_image_brightness` | contínuo (0 a 1) | Brilho/iluminação da foto via OpenCV |
| `photo_image_contrast` | contínuo (0 a 1) | Contraste visual da foto |
| `photo_image_sharpness` | contínuo (0 a 1) | Nitidez estimada por variância do Laplaciano |
| `photo_image_colorfulness` | contínuo (0 a 1) | Vivacidade/cor da imagem |
| `photo_body_width_bucket_narrow` | binário | Silhueta visual estreita, só com sinal corporal confiável |
| `photo_body_width_bucket_medium` | binário | Silhueta visual média, só com sinal corporal confiável |
| `photo_body_width_bucket_wide` | binário | Silhueta visual ampla, só com sinal corporal confiável |

Toda feature engineering está em `src/features.py:extract_features()`.
Features visuais semânticas leves são extraídas em `src/photo_features.py` e
configuradas por `photos.semantic_analysis`. `emotion_auto_download` deve ficar
`false` por padrão para não tentar baixar pesos durante uma sessão; se o peso de
emoção do DeepFace não existir localmente, `photo_face_smile_score` usa apenas
o fallback leve do OpenCV e fica neutro quando não houver evidência clara.
Use linguagem neutra
("silhueta visual", "expressão/sorriso", "foto nítida/escura"); não trate esses
sinais como medição objetiva de peso, beleza ou BMI.

### 2. Feedback forte de treino

O projeto depende de sinais de feedback explícitos para não misturar preferências
visuais com preferências textuais. Toda IA/agente que alterar prompt, review ou
treino deve preservar estas regras:

- Sempre salve `feedback_domain`, `feedback_reason`, `feedback_intensity` e
  `feedback_sentiment` quando uma decisão humana vira treino real.
- Quando a UI/coleta tiver sinais mais detalhados, preserve os campos antigos e
  salve os novos em `feedback_details` (JSON) e `feedback_secondary`. Não
  remova nem reescreva histórico antigo; schemas novos devem preencher campos
  ausentes com vazio.
- Detalhes finos de foto podem ser múltiplos e mistos: use
  `feedback_details.photo_positive_details` para pontos que o usuário gostou e
  `feedback_details.photo_negative_details` para pontos que o usuário não
  gostou, mesmo que a decisão final seja CURTIR ou NÃO CURTIR.
- O treino visual usa detalhes finos apenas quando alinhados ao alvo visual:
  detalhes positivos reforçam exemplos de CURTIR/`photo_score_adjustment=higher`
  e detalhes negativos reforçam exemplos de NÃO CURTIR/`photo_score_adjustment=lower`.
  Isso evita ensinar que um rosto/corpo elogiado é ruim só porque o perfil foi
  recusado por texto ou descritores.
- Quando o usuário concordar com a decisão geral mas quiser corrigir a nota da
  foto, salve `feedback_details.photo_score_adjustment` como `higher` ou
  `lower`, junto de `photo_score_reason` e `photo_score_intensity`. O treino
  de foto usa esse alvo próprio; o treino textual/decisão geral continua usando
  o label final do perfil.
- A aba de rotulagem visual só de fotos está desabilitada por padrão em
  `config.yaml` (`review_ui.photo_deep_enabled: false`). Se reativada, salva
  registros independentes em `data/photo_deep_feedback.jsonl`; esse dataset
  pode treinar um modelo visual separado em `models/photo_deep_classifier.pkl`,
  mas não deve alterar o ensemble principal sem etapa explícita de
  validação/integração.
- O botão de `Super like` na Review UI é rótulo de treino, não ação automática:
  salva `target_action: "super_like"` em `feedback_details` e mantém o label
  binário como CURTIR. `model_training.py` treina um classificador separado
  `superlike_pipeline` com perfis reais curtidos: classe positiva =
  `target_action == "super_like"`, classe negativa = curtidas normais.
  `predictor.py` expõe `superlike_probability`, mas o swiper ainda não deve
  executar swipe para cima sem pedido explícito.
- Domínios válidos: `photo`, `interests`, `bio`, `descriptors`, `other`.
- Quando `feedback_domain == "photo"`, salve `feedback_reason` canônico com
  um destes valores: `photo_face`, `photo_body`, `photo_context`,
  `photo_style`, `photo_general`.
  - `photo_face`: rosto/traços faciais.
  - `photo_body`: corpo/forma física/atração corporal.
  - `photo_context`: fundo, lugar, ambiente, estilo de vida sugerido pela foto.
  - `photo_style`: pose, roupa, estética ou qualidade da foto.
  - `photo_general`: foto geral/sem certeza.
- `other` significa "outro/sem certeza" e deve ter peso baixo no treino. Não use
  campo vazio como atalho; normalize vazio para `other`.
- Intensidade válida: `1` pouco, `2` médio, `3` muito. Se faltar intensidade,
  use `1` para `other` e `2` para domínios específicos.
- Use `src/feedback.py:normalize_feedback()` para normalizar sinais vindos do
  terminal, UI web ou scripts.
- No swipe interativo, o motivo principal deve ser perguntado após a decisão:
  `[f] foto`, `[i] interesses`, `[b] bio`, `[d] descritores`,
  `[o/Enter] sem certeza`.
- Na revisão pós-sessão, aplicar uma revisão deve exigir um motivo principal;
  pular revisão pode ignorar esse requisito.
- Perfis recusados por filtro absoluto (`Nome masculino`, `Filtro trans`,
  `Nenhum rosto detectado na foto`, nome inválido etc.) não devem entrar na
  Review UI de aprendizado fino. Esses casos são decisão operacional do filtro,
  não exemplos para revisar foto/texto; `review_queue.absolute_filter_reason()`
  centraliza essa exclusão.
- A Review UI deve mostrar o snapshot da leitura feita no momento do swipe
  (`review_queue.ai_snapshot`), não recalcular a opinião com o modelo atual.
  Retreinos posteriores podem mudar a leitura; a revisão pós-sessão precisa
  explicar o que a IA viu/valorizou quando decidiu aquele perfil.
- O treino usa pesos por domínio em `config.yaml:model.feedback_weights`: sinais
  de foto devem pesar mais no modelo visual e pouco no textual; sinais de
  bio/interesses/descritores fazem o inverso; sintéticos e histórico sem motivo
  têm peso reduzido.
- Submotivos de foto também têm multiplicadores em
  `model.feedback_weights.photo_reason_multipliers`: `photo_face` e
  `photo_body` podem pesar mais porque há features visuais específicas para
  esses sinais. `photo_context` ainda deve pesar menos enquanto não houver
  enriquecimento visual de cenário/contexto.
- O aprendizado de tokens em `src/text_preferences.py` só deve aprender
  interesses quando `feedback_domain == "interests"` e palavras de bio quando
  `feedback_domain == "bio"`. Se `feedback_details.selected_interests` ou
  `feedback_details.bio_detail` existirem, aprenda só esses sinais específicos;
  caso contrário, mantenha o fallback histórico usando todos os interesses/bio.
- Correções de descritores podem ser independentes da decisão final: use
  `feedback_details.descriptor_positive_details` para descritores que o usuário
  gosta e `feedback_details.descriptor_negative_details` para descritores que o
  usuário não gosta. `src/text_preferences.py` aprende esses sinais como
  preferência textual/estrutural, sem depender só das regras fixas de
  `features.py`.
- Vetos de leitura textual neutralizam um sinal, não invertem o sentimento.
  Use `feedback_details.interest_not_positive`, `interest_not_negative`,
  `bio_not_positive`, `bio_not_negative`, `descriptor_not_positive` e
  `descriptor_not_negative` quando o usuário marcar que a IA exagerou um sinal
  positivo/negativo. `src/text_preferences.py` deve suprimir ou puxar esse
  termo para neutro nas próximas pontuações, e `src/features.py` deve aplicar
  o mesmo veto aos contadores fixos de bio/interesses/descritores somente após
  repetição suficiente para evitar desligar preferências globais por um clique
  isolado. Preserve o histórico antigo.
- Sinais corporais não devem ser tratados como medida real de peso/BMI. Use os
  campos como proxies visuais de composição da foto: corpo visível, corpo
  inteiro, meio corpo, close-up, largura visual e qualidade do sinal.
- O enriquecimento pós-sessão deve rodar por `python3 src/offline_enrichment.py`
  ou `python3 src/backfill_photo_features.py`. Ele atualiza as features visuais
  das fotos salvas; o comando `offline_enrichment.py` também atualiza
  `data/review_queue.csv` para que a revisão póstuma carregue os novos sinais.
  Depois de perfis reais/revisados entrarem em `profiles.csv`, o retreino permite
  que o modo não interativo use esses padrões nas próximas decisões.
- Explicações de decisão devem continuar agrupadas em resumo, sinais
  pró-curtir, sinais pró-passar, fotos/rosto, corpo/composição, texto/perfil,
  incertezas e features influentes. Use linguagem de "sinal visual" ou
  "largura visual", não alegações de peso/BMI real.
- Probabilidades finais devem passar por `model.probability_safety`: evite
  exibir 95–99% quando foto/texto não concordam fortemente. Se a calibração
  aplicar cap, preserve `raw_probability` apenas para auditoria.
- Após retreino automático, a fila pendente de swipes deve ser recalculada com
  o modelo novo. Não deixe perfis já enfileirados usando probabilidades de um
  modelo anterior.
- A fila de swipes deve ser limitada e deduplicada (`swiper.max_pending_queue`,
  `drop_duplicate_profiles`) para evitar backlog velho quando a extensão enviar
  levas repetidas. Por padrão, não pause swipes/captura só porque a aba perdeu
  foco (`swiper.auto_pause_on_focus_loss: false` e
  `browser.pause_capture_when_unfocused: false`), porque notificações e troca
  de janela podem quebrar o `/current`. F8 só deve pausar quando Tinder/terminal
  estiver em foco; F10 deve continuar global como parada de emergência.
- Quando um swipe não for confirmado após as tentativas configuradas, use
  `swiper.reload_on_swipe_failure: true` para solicitar reload e descartar a
  fila stale, em vez de ficar parado indefinidamente esperando F8.
- O servidor mantém uma fila interna de batches de perfis recebidos da extensão.
  Se a fila atingir `swiper.max_pending_batches` (padrão: 3), batches **antigos**
  são descartados para dar lugar ao novo — priorizando sempre os dados mais
  recentes. Além disso, se a memória ultrapassar `swiper.mem_alert_threshold_percent`,
  **todos** os batches pendentes são descartados imediatamente para evitar
  travamento (via `_clear_pending_batches`). O endpoint `POST /reload-start`
  (enviado pela extensão ao detectar reload da página) também limpa todos os
  batches pendentes para evitar processar perfis de uma sessão anterior.
- **Política de reloads (anti-spam):** o swiper evita recarregar a página do
  Tinder desnecessariamente. Em falha de sincronização (`sync`), primeiro tenta
  **pular o perfil** (até `swiper.max_consecutive_sync_skips`, padrão 3) antes
  de solicitar reload. Só recarrega após N pulos consecutivos, indicando problema
  estrutural na página. Em falha de swipe (`swipe_verify`), o reload também fica
  sujeito a `swiper.min_reload_interval_seconds` (padrão 90s) de cooldown —
  dentro desse intervalo, pausa para intervenção manual em vez de recarregar.
  Um swipe confirmado com sucesso reseta os contadores de falha.
  Parâmetros relevantes: `sync_max_wait_count` (padrão 20 = 10s de tentativas),
  `min_reload_interval_seconds` (padrão 90), `max_consecutive_sync_skips` (padrão 3).
- A Review UI deve preservar foto local para revisão (`photos.save_liked: true`
  e `photos.save_disliked: true`). O `photo_url` em `review_queue.csv` é apenas
  fallback temporário quando a foto local ainda não existe ou falhou; não assuma
  que pendências antigas sem `photo_url` são recuperáveis.
- No swipe automático, `src/photo_features.py:analyze_photos()` escolhe uma
  foto boa para rosto (`_best_face_photo_url`) e outra para corpo/composição
  (`_best_body_photo_url`). `src/photo_storage.py:download_profile_photos_async()`
  salva a de rosto/corpo principal no caminho legado do review e, quando a foto
  corporal for diferente, salva também `*_body.jpg`. Não volte a salvar apenas
  a primeira foto do Tinder sem usar esses seletores.
- A extensão reporta foco/visibilidade por `/browser-state`, mas por padrão o
  servidor não deve pausar captura só por perda de foco
  (`browser.pause_capture_when_unfocused: false`). Se essa config for reativada,
  aí sim ignore `/profiles` e `/current` enquanto `capture_active=false`.
- O Chrome automático é controlado por `config.yaml:browser`. Ele abre um perfil
  persistente em `data/chrome_profile`, carrega a extensão local e navega para
  `https://tinder.com/app/recs`. No primeiro uso desse perfil, o usuário precisa
  fazer login uma vez.

### 3. Modelo

- Texto e foto são treinados como pipelines separados.
- **Poucos dados:** `LogisticRegression` (mais estável).
- **Dados suficientes:** `LightGBM` regularizado, via `LGBMClassifier`.
- O meta-modelo combina probabilidades de texto/foto usando predições
  fora-da-amostra (cross-validation), para evitar vazamento e probabilidades
  infladas.
- Todos ficam dentro de `sklearn.pipeline.Pipeline`; o modelo final é
  serializado em `models/classifier.pkl` via pickle.

### 4. Cold start

Gerador em `src/synthetic.py` consegue criar 80 perfis fictícios, mas o arquivo
`data/synthetic.csv` foi esvaziado para não participar mais do treino. Mantenha
assim salvo pedido explícito de bootstrap sintético.

### 5. Retreino automático

Após cada `model.retrain_every` (padrão: 10) novos perfis reais rotulados, o modelo é retreinado com os dados disponíveis. Hoje o treino efetivo é só com dados reais porque `data/synthetic.csv` não tem linhas.

---

## Configuração (config.yaml)

O usuário edita `config.yaml` para ajustar critérios sem tocar no código:

```yaml
preferences:
  age_range: [18, 30]
  preferred_interests: ["academia", "viagem", ...]
  positive_bio_keywords: ["aventura", "leitura", ...]
  negative_bio_keywords: ["ex", "filhos", ...]
  bio_min_length: 20
  disliked_names: []

model:
  min_samples_for_rf: 30
  retrain_every: 10
```

---

## Formato do dataset CSV

`data/profiles.csv` e `data/synthetic.csv` seguem o mesmo schema:

| Coluna | Tipo | Descrição |
|---|---|---|
| `name` | string | Nome do perfil |
| `age` | int | Idade |
| `bio` | string | Texto da bio |
| `interests` | string | Interesses separados por vírgula |
| `label` | int (0 ou 1) | 0 = não curtir, 1 = curtir |
| `source` | string | `"real"` ou `"synthetic"` |

---

## Como adicionar novas features

1. Adicione a lógica de extração em `src/features.py:extract_features()` — retorne o valor no dict
2. Adicione o nome em `src/features.py:FEATURE_NAMES` (ordem importa — é a ordem do vetor X)
3. Adicione o rótulo legível em `src/features.py:FEATURE_LABELS`
4. Delete `models/classifier.pkl` para forçar retreino com as novas features
5. Se a feature precisar de config, adicione em `config.yaml` e leia via `prefs`

---

## Decisões de design importantes

- **Sem API do Tinder:** o sistema usa entrada manual para evitar violação de ToS e risco de ban.
- **Dados sintéticos são apenas bootstrap:** nunca devem substituir dados reais do usuário. Se o usuário tiver dados suficientes, os sintéticos se tornam irrelevantes.
- **Random Forest foi escolhido** sobre outros algoritmos pela `feature_importances_` nativa, que alimenta a explicabilidade sem bibliotecas extras.
- **TextBlob para sentimento:** funciona melhor com inglês mas capta o tom geral do texto em português. Para produção, considerar `transformers` com modelo PT-BR.
- **`src/` no sys.path:** o `cli.py` adiciona `src/` ao `sys.path` para que imports relativos funcionem ao rodar `python src/cli.py` da raiz do projeto.

---

## O que NÃO fazer

- Não conectar ao Tinder via API não oficial (viola ToS, risco de ban)
- Não remover os dados sintéticos enquanto o usuário tiver < 30 perfis reais
- Não trocar pickle por outro formato de serialização sem migrar o arquivo existente
- Não mudar a ordem de `FEATURE_NAMES` sem deletar `classifier.pkl` (vetor X incompatível)

---

## Glossário de features — o que cada grupo significa

Este glossário é para relembrar o usuário (iniciante em ML) do que cada grupo de features faz e por que existe.

### Pose (MediaPipe) — `photo_pose_*`

O MediaPipe (biblioteca do Google) detecta 33 pontos anatômicos no corpo (ombros, quadril, cotovelos, joelhos…) a partir de uma foto corporal. A partir desses pontos, o sistema mede:

| Feature | O que representa |
|---|---|
| `photo_pose_shoulder_width` | Largura visual dos ombros relativa à largura da imagem |
| `photo_pose_hip_width` | Largura visual do quadril relativa à imagem |
| `photo_pose_shoulder_hip_ratio` | Proporção ombro/quadril — captura silhueta ampulheta vs. retangular |
| `photo_pose_torso_visibility` | Quão claramente o torso aparece na foto (0 = oculto, 1 = totalmente visível) |
| `photo_pose_torso_height` | Altura do torso na imagem |
| `photo_pose_upper_body_ratio` | Fração da imagem ocupada pela parte superior do corpo |
| `photo_pose_leg_ratio` | Fração da imagem ocupada pelas pernas |
| `photo_pose_body_coverage` | Cobertura geral do corpo na foto |

**Por que importa:** permite que o modelo aprenda preferências de composição corporal sem depender do rótulo manual de "estreita/média/ampla". Funciona mesmo quando o usuário não avaliou aquele perfil na aba de treino corporal.

**Limitação:** só funciona quando o corpo está visível na foto e a detecção MediaPipe consegue localizar os landmarks. Fotos de close-up, grupo ou pose oblíqua podem falhar (torso_visibility ≈ 0).

---

### Embeddings PCA — `photo_emb_pc_01` … `photo_emb_pc_16`

O DeepFace com Facenet512 analisa cada foto de rosto e gera um vetor de **512 números** que descreve matematicamente a aparência do rosto (geometria, expressão, iluminação, estilo). Esses 512 números são abstratos — não dizem "olhos grandes" ou "nariz pequeno", mas codificam padrões que o modelo Facenet512 aprendeu em milhões de rostos.

**O problema:** 512 features por foto são demais para o modelo ML aprender com os ~2000 perfis disponíveis. Haveria overfitting.

**A solução — PCA:** PCA (Análise de Componentes Principais) comprime os 512 números em **16 componentes** que capturam as variações mais importantes entre os rostos do dataset. É como resumir um livro de 512 páginas em 16 capítulos-chave. O modelo não sabe o que cada componente representa — ele aprende sozinho quais correlacionam com os seus likes.

**Por que importa:** permite que o modelo capture padrões de preferência facial que vão além de "tem rosto" ou "foto nítida" — aspectos sutis de aparência que seriam impossíveis de descrever manualmente.

**Limitação:** o PCA precisa de ~30+ rostos para ser treinado e melhora com mais dados. Perfis sem foto de rosto salva recebem NaN nessas features, e o LightGBM os ignora automaticamente.

**Cobertura atual:** ~55% dos perfis têm embeddings PCA preenchidos.

---

## Backlog pós-treino

Quando a fase atual de treinamento do modelo estiver mais madura, implementar:

- Resumo dos traços de foto que mais aparecem nos likes
- Resumo dos interesses e padrões de bio que mais batem com o histórico do usuário
- Taxa de correção manual da IA no modo interativo
- Resumo diário/sessão com perfis mais curtidos, erros mais comuns e sinais aprendidos

---

## Feature: Super Like automático

Swipe para cima (super like) quando o modelo tem alta confiança:
- `probability >= 0.80`
- `text_score >= 0.70`
- `photo_score >= 0.70`
- distância dentro de `super_like.distance_range_km`
- botão de Super Like reportado como disponível pela extensão

Config principal: `super_like` em `config.yaml`.

---

## Melhorias de ML para implementar no futuro (LEMBRAR AO USUÁRIO)

**Importante:** ao iniciar nova conversa, mencione estas melhorias pendentes se o usuário perguntar sobre qualidade do modelo.

### 1. Pairwise Ranking (Bradley-Terry)

Em vez de classificar cada perfil isoladamente (like/pass), treinar um modelo que compara pares: "dado perfil A e perfil B, qual você preferiria?". Isso captura nuances de preferência relativa que classificação binária perde.

- Biblioteca: `lightgbm` com `objective="lambdarank"` ou `sklearn-contrib-py-earth`
- Geração de pares: combinar perfis curtidos vs. passados no mesmo dia/sessão
- Benefício principal: melhor calibração do ranking quando há muitos likes

Arquivos a modificar: `src/model_training.py`, `src/predictor.py`

### 2. Embeddings de bio (NLP semântico)

Usar modelo de sentence embeddings em português para transformar a bio em vetor denso, em vez de depender apenas de palavras-chave e comprimento de texto.

- Modelo sugerido: `neuralmind/bert-base-portuguese-cased` via `sentence-transformers`
- Alternativa leve: `paraphrase-multilingual-MiniLM-L12-v2` (funciona bem em PT-BR)
- Os embeddings viram features adicionais no text pipeline
- Benefício: captura significado semântico ("adoro natureza" ≈ "amo trilhas") em vez de match exato de palavras

Arquivos a modificar: `src/features.py` (nova função `bio_embedding_features`), `src/model_training.py`

### 3. Modelo de visão real para fotos (CLIP / MobileNet)

Substituir as features manuais de foto (sharpness, brightness, face_count etc.) por embeddings extraídos de um modelo de visão pré-treinado.

- Opção A: `openai/clip-vit-base-patch32` — embedding semântico, captura estilo/contexto
- Opção B: `MobileNetV3` — mais leve, bom para atributos visuais baixo nível
- Os embeddings viram features do photo pipeline (redução de dimensão via PCA antes de entrar no LightGBM)
- Benefício: elimina dependência de heurísticas manuais de foto

Arquivos a modificar: `src/photo_analysis.py` (nova função `extract_deep_features`), `src/features.py`, `src/model_training.py`

### 4. Calibração de probabilidade (Platt Scaling)

O LightGBM e o ensemble atual retornam scores, não probabilidades bem calibradas. `abs(prob - 0.5) < 0.1` pode não indicar incerteza real — o modelo pode ser sistematicamente confiante demais ou de menos.

- Implementação: `sklearn.calibration.CalibratedClassifierCV` com `method="isotonic"` ou `"sigmoid"` (Platt scaling)
- Aplicar sobre o meta-modelo após o cross-val OOF
- Benefício duplo: Active Learning fica mais preciso (incerteza = incerteza real) e os limiares de confiança para Super Like ficam mais confiáveis

Arquivos a modificar: `src/model_training.py` (envolver meta-modelo em `CalibratedClassifierCV`), `src/predictor.py`

---

## Opção B: MediaPipe Segmentation para medição corporal por pixel (IMPLEMENTAR QUANDO SOLICITADO)

Alternativa mais precisa ao estimador de silhueta atual (Canny + pose landmarks) para perfis em que o corpo inteiro aparece na foto.

**O que faz:** usa `mp.tasks.vision.ImageSegmenter` (máscara de segmentação semântica) para isolar os pixels do corpo humano na foto e medir largura real do tronco em regiões específicas (ombros, cintura, quadril).

**Por que é melhor que landmarks:** os landmarks de pose estimam pontos anatômicos mas não medem largura real — um ombro largo e um estreito podem ter o mesmo ponto landmark se a pose variar. A máscara de segmentação mede os pixels reais do contorno.

**Modelo sugerido:** `selfie_multiclass_256x256` — segmenta corpo humano em classes (fundo, cabelo, corpo, rosto, roupa). Baixar de `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`.

**Features novas que isso produziria:**
- `photo_shoulder_width_px_ratio` — largura dos ombros em pixels / largura da imagem
- `photo_waist_width_px_ratio` — largura da cintura em pixels / largura da imagem
- `photo_hip_width_px_ratio` — largura do quadril em pixels / largura da imagem
- `photo_shoulder_waist_ratio` — ombro/cintura (hourglass index)
- `photo_segment_body_coverage` — % de pixels classificados como corpo (filtra close-ups)

**Pré-requisito:** foto deve ter corpo visível (`body_visible > 0.3`) e altura suficiente para ver tronco inteiro; filtrar com `body_entire_body` ou `body_half_body` > 0.

**Onde implementar:** `src/photo_features.py` — novo helper `_estimate_segmentation_features(img_bgr)`, chamado dentro de `analyze_photos()` quando `body_visible` passa o limiar. Features salvas no cache JSON como todas as outras.

**Usar linguagem neutra:** chamar de "largura visual dos ombros/cintura" — nunca BMI, peso, tamanho real.

---

## Plano: Melhoria do modelo de visão (IMPLEMENTAR — leia tudo antes de começar)

**Prioridade: ALTA. Apresente este plano ao usuário no início da próxima conversa.**

### Contexto do sistema atual

O sistema já usa **DeepFace com Facenet512** (modelo de 512 dimensões) para cada foto de rosto:

- `src/photo_features.py` — baixa a foto, roda `DeepFace.represent(model_name="Facenet512")`, salva o embedding bruto (`_embedding`) no cache `data/photo_feature_cache.json`
- `src/face_embeddings.py` — mantém a média (centroide) das faces curtidas/passadas em `data/face_preference.pkl`; computa similaridade cosseno → feature `photo_face_similarity` (escalar único 0–1)
- O embedding bruto de 512 dimensões **já é calculado e está no cache** — mas é descartado depois de virar um único número

**O problema:** o modelo ML recebe apenas `photo_face_similarity` (1 número). Os outros 511 números do embedding — que codificam geometria do rosto, expressão, iluminação, estilo — são jogados fora. Isso é a maior perda de informação do sistema.

**A solução:** usar PCA para comprimir os 512 números em ~16–32 componentes principais e incluí-los como features no modelo. **Custo extra em swipe: zero** — o embedding já é computado, só precisamos aproveitá-lo.

---

### Arquitetura da solução

```
Situação atual:
  foto → DeepFace → embedding[512] → cosine_sim → photo_face_similarity (1 feature)

Situação nova:
  foto → DeepFace → embedding[512] → PCA(16) → photo_emb_pc_01..16 (16 features)
                                   ↓
                              (também mantém photo_face_similarity — não remove)
```

---

### Plano de implementação (passo a passo)

#### Fase 1 — `src/photo_embedding_pca.py` (arquivo novo)

Criar módulo responsável por PCA dos embeddings. Deve conter:

```python
# Caminho onde salvar o modelo PCA treinado (junto com o modelo ML)
PCA_PATH = ROOT_DIR / "data" / "photo_embedding_pca.pkl"
N_COMPONENTS = 16  # ajustar se necessário

def fit_pca(embeddings: list[np.ndarray]) -> PCA:
    """Treina PCA nos embeddings disponíveis. Salva em PCA_PATH."""

def load_pca() -> PCA | None:
    """Carrega PCA salvo. Retorna None se não existir."""

def embedding_to_features(embedding: np.ndarray | None, pca: PCA | None) -> dict:
    """
    Recebe embedding[512], retorna {'photo_emb_pc_01': float, ..., 'photo_emb_pc_16': float}.
    Se embedding ou pca for None → retorna dict com NaN (LightGBM lida nativamente).
    """

def get_embeddings_from_cache() -> dict[str, np.ndarray]:
    """
    Lê data/photo_feature_cache.json e retorna {cache_key: embedding_array}.
    Ignora entradas sem _embedding ou com _embedding=None.
    """
```

**Importante:** os nomes das features (`photo_emb_pc_01` etc.) devem ser constantes exportadas para outros módulos usarem.

---

#### Fase 2 — Adicionar features ao schema

Em `src/features.py`:
- Importar `EMBEDDING_FEATURE_NAMES` de `photo_embedding_pca.py`
- Adicionar ao final de `PHOTO_FEATURE_NAMES`
- Em `extract_features()`, aceitar os valores de `photo_emb_pc_01..16` no dict `photo` (eles virão de `_photo_features` como todos os outros)

Em `src/dataset.py` → `CSV_FIELDNAMES`:
- Adicionar `photo_emb_pc_01` ... `photo_emb_pc_16` ao final da lista

**Regra de backward compatibility:** campos novos ausentes em linhas antigas do CSV ficam vazios → pandas lê como NaN → LightGBM trata como missing → comportamento é "usar as outras features". Nenhum dado antigo é perdido.

---

#### Fase 3 — Integração no treino (`src/model_training.py`)

No início de `train_model()` ou em `_build_X()`:

```python
from photo_embedding_pca import fit_pca, embedding_to_features, get_embeddings_from_cache, N_COMPONENTS

# 1. Carrega todos os embeddings disponíveis no cache
cache_embeddings = get_embeddings_from_cache()  # {cache_key: np.ndarray}

# 2. Para cada row do DataFrame de treino, busca embedding pelo photo_url
#    usando a mesma função _cache_key() de photo_features.py
# 3. Coleta os embeddings encontrados, treina PCA (fit apenas no train set, não no test)
# 4. Salva PCA em PCA_PATH
# 5. Aplica PCA → adiciona colunas photo_emb_pc_01..16 no DataFrame
#    Para linhas sem embedding → NaN (ok)
```

**Atenção:** o PCA deve ser treinado **apenas nos dados de treino** (não no conjunto de validação) para evitar data leakage.

---

#### Fase 4 — Integração na inferência (`src/photo_features.py` ou `src/predictor.py`)

Quando o sistema faz swipe automático:

```python
from photo_embedding_pca import load_pca, embedding_to_features

pca = load_pca()  # carregado uma vez, em memória

# Após DeepFace já computar o embedding (já acontece hoje):
embedding = features.get("_embedding")
emb_features = embedding_to_features(embedding, pca)
features.update(emb_features)
# Agora features tem photo_emb_pc_01..16 junto com as outras
```

Se `pca` for None (modelo ainda não treinado) → `embedding_to_features` retorna NaN → nenhum crash.

---

#### Fase 5 — Script de backfill (opcional mas recomendado)

Criar `src/backfill_embeddings.py` que:
1. Lê `data/photo_feature_cache.json`
2. Para entradas sem `_embedding`, tenta recomputar se a foto ainda estiver salva localmente em `data/photos/`
3. Atualiza o cache
4. Reporta quantos embeddings foram recuperados

Isso permite aproveitar fotos antigas que já estão no disco mesmo que a URL do Tinder tenha expirado.

---

### Dados preservados / perdidos

| Situação | O que acontece |
|----------|----------------|
| Perfis antigos no profiles.csv (sem embedding no cache) | `photo_emb_pc_*` = NaN → LightGBM ignora essas features para eles; continuam contribuindo pelas outras features. **Nenhum dado perdido.** |
| Perfis novos (foto no cache com `_embedding`) | Recebem todas as 16 features → contribuem mais para o aprendizado visual |
| PCA não treinado ainda (< ~30 perfis com embedding) | `load_pca()` retorna None → `embedding_to_features` retorna NaN → comportamento igual ao atual |
| Retreino após a mudança | Treina PCA nos embeddings disponíveis, salva junto com o modelo. As 16 features começam a aparecer no relatório de importância |

---

### Performance em swipe automático

**Não há custo extra.** O `DeepFace.represent(Facenet512)` já roda hoje para cada foto (é o que gera `photo_face_similarity`). Adicionar PCA depois é `< 1ms`. O único custo é memória RAM para manter o array PCA carregado (~16KB).

Se quiser reduzir o custo do DeepFace em si (que já existe hoje), isso é um problema separado — ver seção "Performance do DeepFace" abaixo.

---

### Performance do DeepFace (problema separado, não bloqueia este plano)

DeepFace com Facenet512 demora ~300–800ms por foto em CPU. Isso já ocorre hoje. Se o usuário reclamar de lentidão no swipe automático:

1. **Trocar backend do detector:** `"skip"` é o mais rápido (sem detecção, usa imagem inteira) — já usa `enforce_detection=False`; pode testar `detector_backend="skip"` explicitamente
2. **Trocar modelo:** `VGG-Face` é mais leve que Facenet512, porém embeddings são incompatíveis (precisaria recomputar o centroide em `face_preference.pkl`)
3. **Rodar DeepFace em thread separada** com timeout — já existe lógica de timeout no código, verificar se está ativo

---

### O que NÃO fazer

- **Não trocar para CLIP ou MobileNet** agora — DeepFace/Facenet512 já está funcionando, instalado e gerando embeddings. Adicionar outro modelo = mais dependências, mais RAM, mais tempo de setup sem ganho claro para este caso de uso (fotos de rosto de perfis de dating)
- **Não remover `photo_face_similarity`** — é compatível com todos os dados antigos; manter como feature junto com os PCs
- **Não tentar salvar os 512 componentes brutos no profiles.csv** — CSV ficaria enorme; a ideia é usar PCA para reduzir a 16 antes de salvar qualquer coisa no CSV de treino

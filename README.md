# Match Analyst

Aplicação local de análise de futebol em Python, com interface em inglês e
relatórios determinísticos em português. **E1 a E6 implementadas:** mapas,
estatísticas, xG StatsBomb, modelo próprio, filtros e exportações. Sem serviços
pagos, geração por LLM ou treinamento durante atualizações da interface.

## Abrir a aplicação

Requer Python 3.11 ou superior. Ambiente verificado: Python 3.11.14, macOS ARM64.
Na raiz do projeto:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m streamlit run app.py
```

Abra [http://127.0.0.1:8501](http://127.0.0.1:8501).
No Windows, ative com `.venv\Scripts\activate`.
As dependências são declaradas em `pyproject.toml`; `requirements-lock.txt`
registra as versões verificadas. Para repetir essas versões em ambiente compatível:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Nesta máquina, os dados e o modelo já foram preparados. Em uma instalação nova,
o painel funciona imediatamente com xG StatsBomb após baixar a partida. O modelo
próprio só aparece após a preparação e o treinamento explícitos descritos abaixo.

## Partidas, recortes e exportação

A demonstração é **Brasil 2 × 0 Sérvia, 24/11/2022**, partida `3857258`, descoberta
no catálogo: competição `43`, temporada `106`. São 23 finalizações brasileiras e
5 sérvias. A seleção permite outras competições, temporadas e partidas presentes
no catálogo real; seus arquivos só são baixados quando solicitados.

Os filtros de time, período e jogador atualizam indicadores, mapas, tabela de
chutes, curva acumulada e texto no mesmo recorte. O placar **Final score** é
contexto da partida inteira; **Goals in scope** conta gols no recorte.

Há mapas de chutes, passes e ações defensivas, mapa de calor das origens dos
passes e xG acumulado. A tabela de chutes permite consultar a probabilidade de
cada finalização. Com o artefato salvo, é possível escolher **xG StatsBomb** ou
**Own xG**, identificar a versão e comparar as duas previsões no mesmo chute.
O painel identifica partidas de treino, validação, teste ou fora do conjunto.
A capacidade de generalização do modelo depende do recorte documentado.

Os downloads incluem fatos JSON, estatísticas CSV, chutes CSV, resumo Markdown
e um ZIP com esses arquivos, relatório estruturado, gráficos PNG e registro
separado para futura correção humana. As evidências de cada frase podem ser
consultadas no painel e em `report.json`. `human_review.json` começa com a
correção aprovada ausente; texto gerado e revisão humana permanecem separados.

## Cache e uso sem rede

O primeiro carregamento de cada arquivo requer internet. Depois, ative
**Offline mode**, ou inicie diretamente com:

```bash
MATCH_ANALYST_OFFLINE=1 python -m streamlit run app.py
```

Para preparar apenas a demonstração ou verificar seu cache:

```bash
python scripts/fetch_demo.py
python scripts/fetch_demo.py --offline
```

`MATCH_ANALYST_CACHE_DIR` altera o cache da interface. Os scripts aceitam
`--cache-dir`. `MATCH_ANALYST_MODEL_DIR` altera a pasta do modelo carregado
(padrão: `models/xg_v1`). Um modelo ausente ou inválido é informado claramente;
a fonte StatsBomb continua disponível.

O cache preserva os JSONs brutos. Cada `.meta.json` registra origem, instante UTC
do download, SHA-256, revisão e IDs aplicáveis. A revisão é fixada no primeiro
acesso em `data/raw/source.json`; a configuração de treinamento fixa explicitamente
`4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`.

```text
data/raw/source.json
data/raw/<SHA>/competitions.json (+ competitions.meta.json)
data/raw/<SHA>/matches/<competition>/<season>.json (+ .meta.json)
data/raw/<SHA>/events/<match>.json (+ .meta.json)
```

Arquivos existentes são preferidos mesmo online e têm o checksum conferido.
Corrupção ou cache incompleto geram erro explícito. Para baixar novamente um
arquivo, remova somente seu JSON e o `.meta.json` correspondente e carregue
online. Para iniciar uma revisão atualizada, use um novo diretório de cache ou
um SHA explícito no cliente Python. Não são baixados escalações ou dados 360.

## Preparar e treinar o modelo

A configuração editável fica em `config/dataset.json`. Antes do download em
volume, o comando de catálogo confirma contagens e datas:

```bash
python scripts/prepare_dataset.py --catalog-only
python scripts/prepare_dataset.py --workers 4
python scripts/train_xg.py --model-version logreg-xg-v1
```

O recorte contém seleções masculinas adultas em três torneios, com todas as
partidas encontradas nas listas verificadas:

- **Treino:** Copa de 2018, 64 partidas, 1.667 chutes, 157 gols em finalizações.
  Sem pênaltis: 1.638 chutes e 135 gols; pênaltis: 29 cobranças e 22 gols.
- **Validação:** Euro 2020, realizada em 2021, 51 partidas, 1.251 chutes,
  131 gols em finalizações. Sem pênaltis: 1.234 chutes e 122 gols.
- **Teste:** Copa de 2022, 64 partidas, 1.453 chutes, 169 gols em finalizações.
  Sem pênaltis: 1.430 chutes e 152 gols. Brasil × Sérvia está apenas neste grupo.

Gols contra ficam fora dos alvos por chute e são tratados separadamente no
placar. As contagens acima não incluem cobranças de disputa. O recorte representa
três torneios específicos, não todo o período de 2015 a 2022 nem futebol de clubes,
feminino ou de base. Qualificação, geografia e diferenças entre torneios limitam
a generalização.

A preparação incremental ocupa aproximadamente 531 MiB de dados brutos e gera
`data/processed/manifest.json`, `shots.csv` e `diagnostics.json`. O manifesto fixa
partidas, divisões, configuração e hashes; não há sobreposição de partidas.
O identificador verificado é **`statsbomb-36532832efe3947fefe5`**. Uma segunda
preparação offline produziu o mesmo ID, as mesmas divisões e CSV idêntico byte a
byte. Há 4.371 finalizações elegíveis e 118 cobranças de disputa excluídas.
Nenhum atributo inicial estava ausente neste recorte; ausências futuras são
tratadas pelo pipeline e registradas no diagnóstico.

```bash
python scripts/prepare_dataset.py --offline --output-dir /tmp/match-analyst-replay
```

O modelo usa distância ao centro do gol, ângulo entre os postes, parte do corpo e
tipo de finalização. Distância está em **unidades StatsBomb**; ângulo, em radianos.
Imputação, escala, codificação e regressão são ajustadas somente no treino.
Resultado, destino da bola, ação do goleiro e xG da fonte não entram nos atributos.
A especificação geométrica e os casos extremos estão em
[docs/MODEL_DESIGN.md](docs/MODEL_DESIGN.md).

A menor log loss da validação selecionou **C=1** entre `0.1, 1, 10`.
`selection.json` foi gravado antes da avaliação final no teste. O treino usa
semente 42 e nenhuma ponderação artificial de classes. A referência prevê a
taxa de gols sem pênalti observada no treino.

Resultados da versão **`logreg-xg-v1`**, para finalizações sem pênalti:

- Validação: log loss **0,283298**, referência **0,324273**;
  Brier **0,079391**; ROC-AUC **0,752016**.
- Teste: log loss **0,289278**, referência **0,342175**;
  Brier **0,083531**; ROC-AUC **0,780550**.
- StatsBomb no teste: log loss **0,266505**, Brier **0,076192**;
  ROC-AUC **0,814775**. Essa comparação não participou da seleção.

O modelo melhorou a referência simples nesse recorte e teve métricas inferiores
às da StatsBomb no teste. Esses resultados não estabelecem desempenho em outras
populações. O gráfico `models/xg_v1/calibration.png` e os pontos/contagens por
intervalo nos metadados permitem examinar a calibração das probabilidades.

O componente separado de pênaltis usa somente as 29 cobranças de treino:
com 22 gols e prior Beta(1,1), a média posterior é **23/31 = 0,741935**.
`npxg_model` contém apenas finalizações sem pênalti; `xg_model` acrescenta essa
estimativa aos pênaltis durante o jogo. Métricas de npxG, pênaltis e total são
registradas separadamente para cada divisão.

O treinamento salva pipeline completo, metadados, seleção congelada, cópias dos
manifestos, versões das dependências, configuração, datas, métricas e checksums.
O comando recusa sobrescrever uma versão existente. Para uma nova versão use
uma pasta nova; decisões influenciadas pelo teste já aberto exigem uma futura
avaliação independente. A interface apenas carrega e prevê; ela nunca treina.

## Executar análises sem a interface

```bash
python scripts/analyze_match.py --offline
python scripts/analyze_match.py --offline --xg-source xg_model
python scripts/analyze_match.py --offline --match-id 3869685 --xg-source xg_model
python scripts/analyze_match.py --offline --team-id 781 --period 2
```

O script usa o mesmo núcleo de métricas, narrativa e gráficos e salva o ZIP e os
arquivos extraídos em `outputs/`. O exemplo `3869685` é a final Argentina × França,
confirmada pelo catálogo, com 3–3 após prorrogação. As 30 finalizações normais,
três pênaltis de jogo e oito cobranças do desempate exercitam regras diferentes;
os 21 eventos do período 5 são excluídos da análise normal.

## Definições das métricas

- **Finalizações:** eventos Shot válidos dos períodos 1–4. Dentro da área:
  `x >= 102` e `18 <= y <= 62`; fora: demais origens válidas.
- **Gols no recorte:** finalizações com resultado Goal e gols contra creditados
  ao adversário, com representações espelhadas reconciliadas. Não se inventa
  uma finalização para um gol contra. A reconciliação com o placar usa o jogo
  completo antes dos filtros e registra divergências explicitamente.
- **Passes tentados:** eventos Pass válidos. **Completos:** chave `pass.outcome`
  ausente no JSON. A taxa é completos/tentados no mesmo recorte; no CSV é fração.
- **Entradas por passe no terço final:** passes completos com origem `x < 80`
  e destino `x >= 80`. Corredor pelo destino: esquerda `y < 80/3`, centro
  `80/3 <= y < 160/3`, direita `y >= 160/3`, na orientação de ataque da equipe.
- **Defesa:** pressões, interceptações, duelos Tackle e cortes ficam separados.
  A soma não é chamada de desarmes.
- **xG:** soma de uma única fonte identificada. **npxG:** mesma fonte, excluindo
  pênaltis de jogo. **xG/chute:** total dividido pelas finalizações do recorte.
  Um valor ausente impede apresentar um total completo; cobertura é informada.
- **Sem denominador:** valor indisponível. Zero chutes permanece zero chutes.
  Passes sem coordenadas continuam no denominador de acerto, mas não em métricas
  espaciais; a cobertura fica registrada.

## Coordenadas, tempo e narrativa

O campo mantém unidades StatsBomb `0–120 × 0–80`, com y crescendo de cima para
baixo. Cada equipe ataca à direita nos seus mapas; não há inversão automática
por visitante ou tempo. Mapas defensivos também preservam o referencial do time
que agiu. O heatmap conta origens dos passes, não permanência de jogadores;
os dois times compartilham a mesma escala de volume.

Eventos são ordenados por período e índice; IDs repetidos mantêm a primeira
ocorrência. Coordenadas inválidas e demais exclusões têm motivos auditáveis.
Períodos 1–4 incluem prorrogação; período 5 é sempre separado. A curva acumulada
mantém os acréscimos de cada período, sem intervalos de descanso. A referência
de tempo usa todos os eventos da partida, mesmo com time/jogador filtrado;
marcadores de gols conservam rótulos do relógio original.

`generate_report(facts) -> report` é a interface para o futuro gerador local.
O objeto de fatos contém partida, filtros, métricas/denominadores, origem e versão
do xG, comparações, limitações e evidências. Cada frase referencia suas chaves
em `report.sentences[].evidence_keys`. O texto não calcula estatísticas.

A expressão de concentração em um corredor exige pelo menos 10 entradas e uma
vantagem de 10 pontos percentuais sobre o segundo corredor. Esses limites podem
ser alterados nos argumentos `min_entries` e `corridor_margin`. Não há inferências
de posse, intensidade, domínio tático ou resultado esperado a partir de uma
única contagem. Fine-tuning e LLM ficam fora desta versão.

## Estrutura e verificações

```text
app.py                           Seleção, filtros, apresentação e downloads
src/match_analyst/
  data.py                        Catálogo/cache/proveniência e leitor Sblocal
  normalize.py                   Eventos e finalizações validados
  metrics.py                     Estatísticas, filtros, fatos e reconciliação
  plots.py                       Mapas, curvas e comparação
  narrative.py                   Relatório determinístico com evidências
  exports.py                     JSON, CSV, Markdown, PNG e ZIP
  dataset.py                     Seleção e base de treinamento reproduzível
  xg.py                          Geometria, pipeline, avaliação e inferência
config/dataset.json              Recorte editável com revisão fixa
scripts/                         Preparação, treinamento e análise explícitos
tests/                           Amostras pequenas, sintéticas e sem rede
data/manifests/                  Snapshots rastreáveis desta entrega
data/raw/ e data/processed/      Dados locais, fora do Git
models/xg_v1/                    Pipeline e metadados locais, fora do Git
outputs/                         Análises exportadas, fora do Git
docs/PROJECT_SPEC.md             Especificação original; escopo ampliado pelo usuário
docs/MODEL_DESIGN.md             Desenho do modelo e contrato de atributos
```

```bash
python -m pytest
python -m pip check
```

**123 testes passaram** na verificação final. A suíte cobre coordenadas/orientação, geometria extrema, duplicados, passes,
gols contra, filtros vazios, separação treino/validação/teste, prevenção de
vazamento, pipeline treinado só no treino, persistência, penalidades, narrativa,
exportações, cache sem rede e interface. O número atualizado de testes aparece
na execução; os testes principais não dependem dos dados reais baixados.

A auditoria offline das 179 partidas encontrou **zero divergências de placar**,
zero IDs duplicados e zero eventos inválidos; 24 partidas tinham gols contra.
Foram separados 292 eventos de disputa no normalizador completo (118 eram
chutes no preparador de finalizações). França × Suíça, da Euro, e a final de 2022
validaram prorrogação e disputa. A base foi reproduzida offline; tabelas, curvas,
fatos, CSV e texto foram confrontados em ambas as fontes e múltiplos filtros.
Snapshots dos manifestos, diagnóstico, avaliação e auditoria ficam versionáveis
em `data/manifests/`.

## Fonte, atribuição e limites

Dados: [StatsBomb Open Data](https://github.com/statsbomb/open-data).
O [README oficial](https://github.com/statsbomb/open-data#terms--conditions)
pede atribuição e logo ao compartilhar ou publicar análises; a interface inclui
ambos e registra o [link da licença](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf)
nos metadados. O logo local tem origem registrada em `assets/README.md`.
A documentação de [tipos de campo do mplsoccer](https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitch_types.html)
registra a geometria adotada.

A aplicação é local e depende da cobertura da fonte; partidas ainda não guardadas
exigem conexão. A regressão é um primeiro modelo pequeno, sem atributos 360,
pressão ou posição dos demais jogadores. Não há treinamento automático nem LLM.
A próxima evolução deve ampliar e avaliar a base com um novo protocolo externo,
melhorar os atributos sem vazamento e reunir revisões humanas aprovadas.

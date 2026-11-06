# Modelo próprio de xG

O componente principal estima **npxG**: probabilidade de gol de uma finalização sem pênalti. Uma regressão logística produz probabilidades; não se usa balanceamento artificial das classes. Um componente separado para pênaltis durante a partida permite publicar `xg_model` com as duas partes. A disputa de pênaltis nunca participa de treino, avaliação ou inferência.

## Base e separação

A configuração inicial usa futebol masculino adulto em torneios de seleções: Copa de 2018 para treino; Euro 2020 disputada em 2021 para validação; Copa de 2022 para teste final. Brasil × Sérvia de 2022 pertence somente ao teste. São 64, 51 e 64 partidas no catálogo confirmado, respectivamente; o manifesto registra a disponibilidade efetivamente preparada e suas exclusões. Essa seleção cobre torneios específicos; não representa ligas, todas as temporadas de 2015–2022 nem todo o futebol masculino. O conjunto é pequeno e sujeito à mudança de distribuição entre torneios.

`manifest.json` fixa a revisão da fonte e os identificadores das partidas por divisão. A tabela de chutes precisa concordar com o manifesto: nenhuma partida pode aparecer em duas divisões. O treinamento verifica o checksum da tabela quando fornecido pelo manifesto e grava novamente os hashes dos insumos no artefato.

## Geometria e atributos permitidos

Os chutes mantêm a convenção StatsBomb: campo de 120 × 80, gol atacado no lado `x=120`, centro `(120,40)` e postes `(120,36)` e `(120,44)`. A distância é euclidiana ao centro, em **unidades StatsBomb**, sem conversão para metros. O ângulo de abertura é `atan2(abs(cross(v1,v2)), dot(v1,v2))`, em radianos, entre os vetores que partem do chute em direção aos postes.

Na linha do gol e entre os postes, o ângulo é π. Exatamente em um poste, um vetor é nulo: convencionamos ângulo zero e registramos essa decisão nos metadados. Coordenadas ausentes, não finitas ou fora dos limites produzem atributos geométricos ausentes; dados normalizados para treino já excluem origens inválidas.

A whitelist de entrada tem somente `distance_statsbomb`, `angle_radians`, `body_part` e `shot_type`. A geometria é recalculada das origens; colunas geométricas pré-calculadas não são aceitas implicitamente. Resultado, gol, destino da bola, ação do goleiro e xG da StatsBomb permanecem fora dos atributos. `prepare_features` descarta todas as demais colunas.

O pipeline persistido inclui construção dos atributos, imputação numérica pela mediana, padronização numérica, imputação categórica com `Unknown`, codificação one-hot com categorias desconhecidas ignoradas e regressão logística. Imputação, escala, vocabulário e coeficientes são ajustados **somente no treino**.

## Seleção e avaliação

São ajustados candidatos `C = 0.1, 1, 10` usando o treino. Escolhe-se a menor log loss na validação; empates exatos favorecem o menor C. `selection.json` é gravado com essa escolha antes de avaliar os rótulos do teste. Não se reajusta o pré-processamento nem o modelo usando a validação depois da seleção. A comparação com xG StatsBomb ocorre somente no relatório final de teste e não participa da seleção.

A referência prevê a taxa de gols sem pênalti observada no treino. O relatório inclui log loss, Brier score, ROC-AUC complementar e pontos de calibração em dez intervalos uniformes. Uma divisão sem duas classes não recebe ROC-AUC. Recortes vazios recebem métricas indisponíveis. Valores ausentes de xG são contabilizados por cobertura, e não substituídos por zero. `calibration.png` mostra validação e teste; é um diagnóstico, não um calibrador ajustado nos dados de avaliação.

As métricas são separadas em `npxg`, `penalties` e `all_shots`, por divisão, cada uma com `model` e `baseline`. Para pênaltis a referência usa a taxa bruta de conversão do treino. Para todos os chutes combina-se a referência sem pênaltis com essa taxa. O relatório registra quando o modelo não supera a referência na validação; nesse caso há uma limitação explícita para orientar investigação de cobertura e atributos.

## Componente de pênaltis

Somente pênaltis dos períodos 1–4 do treino estimam o componente. Com `g` gols em `n` cobranças, uma prior Beta(1,1) fornece média posterior `(g+1)/(n+2)`. Essa suavização evita probabilidades exatamente zero ou um em amostras pequenas; não é uma garantia de calibração. Quantidade, gols, prior, posterior e probabilidade ficam nos metadados. Sem cobranças de treino, o componente fica indisponível; a média da prior não é publicada como uma estimativa aprendida.

`npxg_model` contém apenas previsões sem pênalti. `xg_model` usa a regressão para finalizações sem pênalti e a probabilidade separada para pênaltis elegíveis. Não há mistura com `xg_statsbomb`. Não-chutes e disputa de pênaltis recebem ausência de previsão.

## Execução e artefatos

Depois de preparar a base local:

```sh
python scripts/train_xg.py --dataset-dir data/processed --output-dir models/xg_v1 --seed 42
```

O treinamento é explícito e funciona sem rede. A interface carrega o artefato existente; suas atualizações não treinam modelos. O comando recusa sobrescrever uma seleção ou modelo existente. Para uma nova versão, use uma nova pasta e preserve as versões anteriores. A reutilização do teste para sucessivas decisões humanas deve ser evitada: novas decisões orientadas por esse teste exigem uma avaliação externa futura para uma estimativa independente de generalização.

Cada pasta reúne `model.joblib`, `metadata.json`, `selection.json`, `dataset_manifest.json`, `split_manifests.json` e `calibration.png`. Os metadados incluem versão, atributos, conjunto de dados, revisão da fonte, divisões, contagens de partidas/chutes/gols e competições, métricas, configuração, semente, datas, versões das dependências e checksums. `model_membership` identifica se a partida pertence a treino, validação, teste ou se não consta na base.

`load_model` valida os hashes antes de carregar o pipeline. Joblib utiliza pickle: os artefatos devem ser produzidos localmente e por código confiável; hashes não autenticam um produtor desconhecido.

## Referências técnicas

O desenho segue a documentação oficial de [prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html), [pré-processamento de tipos mistos](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html), [métricas probabilísticas](https://scikit-learn.org/stable/modules/model_evaluation.html), [calibração](https://scikit-learn.org/stable/modules/calibration.html) e [persistência de modelos](https://scikit-learn.org/stable/model_persistence.html). Os eventos vêm de [StatsBomb Open Data](https://github.com/statsbomb/open-data), conforme sua atribuição e condições de uso.

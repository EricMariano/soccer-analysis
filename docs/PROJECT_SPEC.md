Quero construir, gradualmente, uma aplicação local de análise de partidas de futebol em Python. Ela deve combinar visualizações com mplsoccer, estatísticas da partida e um modelo próprio de gols esperados, chamado xG.

Use este plano como especificação do projeto. Implemente por etapas, mantendo cada entrega pequena, funcional e verificável. Nesta primeira execução, faça somente a Etapa 1 descrita adiante.

**1. Objetivo do produto**

O usuário seleciona uma partida e consegue:

- Consultar o placar e as principais estatísticas.
- Visualizar mapas de chutes, passes e ações defensivas.
- Visualizar mapas de calor de ações com bola.
- Comparar o volume e a qualidade das finalizações dos times.
- Consultar o xG de cada chute e o xG acumulado durante a partida.
- Ler um resumo em português fundamentado nas estatísticas calculadas.
- Exportar os resultados da análise.

A aplicação será inicialmente local, para estudo e portfólio, com uma interface simples em inglês.

**2. Decisões iniciais**

Fonte de dados: StatsBomb Open Data.

Visualizações de campo: mplsoccer.

Interface: Streamlit.

Análise de dados: pandas e NumPy.

Modelo de xG: scikit-learn, começando por regressão logística.

Geração de texto inicial: regras e modelos de frases determinísticos.

A geração de texto com um modelo de linguagem leve e um possível fine-tuning será uma evolução posterior. Prepare uma estrutura de dados adequada para isso, mas mantenha a primeira versão independente de modelos de linguagem e de serviços pagos.

O Kloppy poderá ser incorporado quando houver uma segunda fonte de dados. Neste projeto inicial, trabalhe diretamente com o formato da StatsBomb e os leitores do mplsoccer.

**3. Como o sistema deve funcionar**

Fluxo principal:

Dados brutos da StatsBomb → validação e normalização → estatísticas e preparação das finalizações → estimativa de xG → gráficos e fatos estruturados → resumo textual → interface e exportação.

Mantenha as responsabilidades separadas:

- A camada de dados obtém e organiza os arquivos.
- A camada de métricas calcula os números.
- A camada de xG treina o modelo e gera probabilidades.
- A camada de visualização desenha os gráficos.
- A camada de narrativa transforma fatos calculados em texto.
- A interface permite selecionar filtros e consultar os resultados.

Os cálculos devem funcionar fora da interface. Evite concentrar toda a lógica no arquivo do Streamlit.

**4. Fonte, seleção e armazenamento dos dados**

Use o catálogo real da StatsBomb para descobrir competições, temporadas e partidas disponíveis.

Para a primeira demonstração, prefira Brasil x Sérvia na Copa de 2022. Localize e confirme o identificador pelo catálogo, em vez de adivinhar o número.

Baixe inicialmente apenas:

- O catálogo de competições necessário.
- A lista de partidas da competição e temporada escolhidas.
- Os eventos da partida selecionada.
- As escalações, se forem necessárias para os filtros.

Implemente um cache local para reaproveitar os arquivos baixados. Depois do primeiro carregamento, a partida já armazenada deve poder ser analisada sem acesso à internet.

Preserve os arquivos brutos e registre:

- URL de origem.
- Data do download.
- Identificadores da competição, temporada e partida.
- Checksum dos arquivos.
- Revisão do repositório de origem, quando disponível.

Na preparação da base de treinamento, use uma revisão fixada da fonte e um manifesto das partidas. Isso deve permitir reproduzir o conjunto usado em cada versão do modelo.

Registre a atribuição e as condições de uso indicadas pela fonte.

**5. Organização do projeto**

Use um projeto Python com dependências declaradas e instruções claras de instalação.

Uma estrutura inicial possível:

- `app.py`: interface Streamlit.
- `src/match_analyst/data.py`: catálogo, download, cache e leitura.
- `src/match_analyst/normalize.py`: normalização e validação dos eventos.
- `src/match_analyst/metrics.py`: estatísticas e fatos da partida.
- `src/match_analyst/plots.py`: gráficos.
- `src/match_analyst/xg.py`: preparação de atributos, treinamento e inferência.
- `src/match_analyst/narrative.py`: resumo determinístico.
- `scripts/train_xg.py`: execução explícita do treinamento.
- `tests/`: testes relevantes.
- `data/`: dados locais e manifestos.
- `models/`: modelos treinados e seus metadados.
- `outputs/`: resultados exportados.
- `README.md`: instalação, uso e explicação das métricas.

Crie os módulos conforme forem necessários em cada etapa. Prefira funções claras e componentes pequenos. Notebooks podem apoiar explorações, mas o funcionamento da aplicação deve estar no código reutilizável.

**6. Contrato dos dados e regras de futebol**

Preserve identificadores estáveis de partida, evento, time e jogador. Use os nomes para apresentação.

Na tabela normalizada, mantenha os campos necessários para identificar:

- Partida, competição e temporada.
- Time e jogador.
- Período, ordem do evento e tempo.
- Tipo da ação e seu resultado.
- Coordenadas de origem e destino.
- Informações específicas de passes e finalizações.
- xG fornecido pela StatsBomb, quando disponível.

Trate campos opcionais conforme a documentação. Um valor ausente deve receber tratamento explícito; sua interpretação depende da semântica do campo.

Regras obrigatórias:

- Remover ou sinalizar eventos duplicados pelos identificadores.
- Validar coordenadas, tipos e campos essenciais.
- Registrar quantos eventos foram excluídos e por qual motivo.
- Ordenar eventos por período e ordem do evento, considerando acréscimos e prorrogação.
- Diferenciar pênaltis durante o jogo de cobranças na disputa de pênaltis.
- Excluir a disputa de pênaltis das estatísticas normais de chutes, gols e xG.
- Considerar gols contra na reconciliação do placar, sem inventar uma finalização correspondente.

Verifique a convenção de coordenadas da StatsBomb antes de calcular lados e direções. Uma inversão baseada apenas em “visitante” ou “segundo tempo” pode produzir gráficos incorretos.

Ao mostrar os dois times atacando na mesma direção, mantenha a transformação explícita e consistente com o time e o tipo de evento.

**7. Estatísticas da partida**

Implemente inicialmente, por time:

- Total de finalizações.
- Finalizações de dentro e de fora da área.
- Gols no recorte analisado.
- Passes tentados e completos.
- Taxa de acerto de passes.
- Entradas por passe no terço final.
- Distribuição dessas entradas por corredor.
- Contagens de ações defensivas por tipo.
- xG e xG por finalização, quando disponíveis.
- xG sem pênaltis, chamado npxG, quando disponível.

Cada métrica precisa ter uma definição documentada.

Por exemplo:

- “Entradas por passe no terço final”: passes completos que começam fora do terço final e terminam dentro dele.
- “Corredor da entrada”: classificação baseada na coordenada de destino do passe.
- “Taxa de acerto”: passes completos divididos pelos passes tentados no mesmo recorte.

Use os limites da área e dos setores de acordo com o sistema de coordenadas adotado.

Nas ações defensivas, apresente separadamente pressões, interceptações, desarmes identificáveis no formato da fonte e cortes. Evite transformar a soma de ações diferentes em uma contagem chamada “desarmes”.

Um mapa de eventos representa onde aquelas ações ocorreram. Ele não representa, por si só, o tempo que os jogadores permaneceram em cada região.

Para divisões sem denominador, apresente “não disponível”. Um filtro sem finalizações deve mostrar zero chutes e uma explicação adequada.

**8. Estratégia de xG em duas fases**

Primeiro, integre o xG já fornecido pela StatsBomb para validar o painel.

Identifique claramente essa métrica como “xG StatsBomb”. Ela permitirá conferir mapas, somatórios e o gráfico acumulado enquanto o modelo próprio ainda está em desenvolvimento.

Depois, treine e integre o nosso modelo.

Use nomes separados para as previsões:

- `xg_statsbomb`: valor da fonte.
- `xg_model`: previsão do modelo próprio.

O usuário deve conseguir identificar qual origem está sendo exibida. Um total ou gráfico acumulado deve usar uma única origem de xG.

Quando o modelo próprio ainda não estiver disponível, apresente essa condição explicitamente.

**9. Definição do modelo próprio**

O modelo estima a probabilidade de uma finalização resultar em gol, considerando informações disponíveis no momento do chute.

Cada exemplo de treinamento corresponde a uma finalização.

Variável alvo:

- 1 para gol.
- 0 para os demais resultados de finalização válidos.

Comece pelo modelo de finalizações sem pênalti.

Atributos iniciais:

- Distância ao centro do gol.
- Ângulo de abertura do gol.
- Parte do corpo utilizada.
- Contexto ou tipo de finalização, conforme os campos disponíveis.

Pressão e finalização de primeira podem entrar depois, se houver cobertura e semântica consistentes na base selecionada.

Calcule distância e ângulo em uma convenção geométrica documentada. No MVP, a distância pode ficar em unidades StatsBomb, com esse nome explícito. Uma apresentação em metros exige uma conversão coerente e documentada.

O ângulo precisa tratar posições próximas à linha do gol e outros casos extremos sem gerar valores inválidos.

Use um pipeline que reúna o tratamento de valores ausentes, a codificação das categorias, a transformação dos atributos numéricos e a regressão logística.

Informações como resultado do chute, destino final da bola, defesa do goleiro e xG da StatsBomb não podem entrar nos atributos de entrada. Elas introduziriam informação posterior à finalização ou incorporariam a previsão de outro modelo.

Comece sem balanceamento artificial das classes. Avalie a qualidade das probabilidades antes de introduzir mudanças.

**10. Treino, validação e avaliação**

Treinar um modelo exige várias partidas. A partida demonstrada na interface deve ficar fora do treino e da seleção do modelo.

Defina um recorte inicial coerente, por exemplo futebol masculino adulto, e documente sua cobertura. O resultado será válido dentro dos limites dessa base.

Configuração inicial proposta:

- Treino: partidas disponíveis do recorte escolhido entre 2015 e o fim de 2020.
- Validação: partidas elegíveis de 2021 até antes da Copa de 2022.
- Teste: Copa de 2022, mantida separada durante o desenvolvimento.

Antes do download em volume, confirme a disponibilidade real desse recorte no catálogo. Salve a seleção em uma configuração editável.

Verifique se algumas temporadas possuem apenas uma seleção de partidas de determinado clube ou jogador. Registre esses vieses e evite apresentar essa amostra como cobertura integral da competição.

Regras de separação:

- Uma partida pertence a uma única divisão.
- Todos os chutes da mesma partida permanecem juntos.
- O pré-processamento é ajustado apenas com o treino.
- A validação orienta as escolhas.
- O teste final mede o desempenho do modelo escolhido.
- A comparação com o xG da StatsBomb não deve orientar a seleção após abrir o teste.

Registre o número de partidas, chutes e gols em cada divisão, além da distribuição por competição.

Avalie:

- Log loss.
- Brier score.
- Curva de calibração.
- ROC-AUC como medida complementar.

Compare a regressão com um modelo simples que prevê a taxa média de gols observada no treino.

A acurácia isolada é insuficiente para avaliar xG. O objetivo é produzir probabilidades úteis, e não apenas classificar quase todos os chutes como “não gol”.

Se o modelo não superar a referência simples, registre o resultado e investigue os dados e atributos. Evite prometer um desempenho antes da avaliação.

**11. Tratamento dos pênaltis**

Inicialmente, apresente o modelo próprio como npxG, deixando seu escopo claro.

Em seguida, adicione um componente separado para pênaltis cobrados durante a partida:

- Estime a taxa de conversão com dados de treino.
- Documente a quantidade de cobranças e o tratamento estatístico adotado.
- Aplique essa estimativa aos pênaltis elegíveis.
- Continue excluindo as cobranças da disputa de pênaltis.

Somente depois desse componente, apresente um total próprio de xG que inclua pênaltis.

Avalie e apresente separadamente a qualidade do modelo sem pênaltis e do componente de pênaltis.

**12. Persistência e rastreabilidade do modelo**

Salve o pipeline completo de pré-processamento e previsão.

Junto dele, registre:

- Versão do modelo.
- Lista de atributos.
- Identificador do conjunto de dados.
- Manifestos das divisões.
- Métricas de validação e teste.
- Versões das dependências.
- Configuração e semente aleatória.
- Data do treinamento.

O painel deve carregar um modelo existente para fazer previsões. O treinamento deve ser uma execução explícita, fora das atualizações da interface.

Marque quando uma partida exibida participou do treinamento. As partidas reservadas para avaliação devem permanecer identificáveis.

**13. Visualizações**

Implemente os seguintes gráficos:

**Mapa de finalizações**

Mostrar posições, resultado do chute e tamanho dos marcadores proporcional ao xG selecionado. Incluir legenda e origem do xG.

**Mapa de passes**

Mostrar origem, destino e resultado. Permitir selecionar time e, depois, jogador.

**Mapa de ações defensivas**

Usar marcadores ou cores distintos para os tipos de ação.

**Mapa de calor**

Começar pelas origens dos passes. Usar o título “Mapa de calor das origens dos passes” para deixar claro o que está sendo contado.

**xG acumulado**

Mostrar uma curva por time, respeitando a ordem dos eventos e identificando períodos, acréscimos e prorrogação.

Todos os gráficos devem ter título, filtros aplicados, unidade ou escala relevante e atribuição da fonte.

Ao comparar mapas de calor, use uma escala comum quando o objetivo for comparar volumes.

**14. Interface**

Organize a interface em quatro áreas simples:

- Seleção da partida e filtros.
- Resumo com placar e indicadores.
- Gráficos da análise.
- Texto e exportações.

Filtros previstos:

- Competição, temporada e partida.
- Time.
- Período.
- Jogador, para visualizações compatíveis.
- Origem do xG, depois da integração do modelo próprio.

Cada componente deve informar seu recorte. O placar final pode continuar visível como contexto, desde que esteja identificado como “placar final”; os indicadores filtrados precisam ter seus próprios rótulos.

Uma mudança de filtro deve atualizar os números, os gráficos e o texto correspondente.

**15. Resumo textual inicial**

Gere um resumo curto em português a partir de um objeto estruturado de fatos.

Esse objeto deve conter:

- Identificação da partida.
- Filtros aplicados.
- Estatísticas com seus denominadores.
- Origem e versão do xG.
- Comparações calculadas.
- Limitações ou dados ausentes.
- Evidências usadas em cada afirmação.

Use modelos de frases como:

“O time A finalizou X vezes, sendo Y de dentro da área.”

“O time B registrou X entradas por passe no terço final; Y ocorreram pelo corredor direito.”

“Pelo modelo selecionado, o time A somou X de xG e o time B, Y.”

Só use expressões como “concentrou as entradas pela direita” quando houver uma regra explícita. Uma configuração inicial pode exigir ao menos dez entradas e uma vantagem de dez pontos percentuais sobre o segundo corredor. Esses limites devem ficar documentados e configuráveis.

Evite inferir posse de bola pela contagem de eventos ou afirmar domínio tático, intensidade de pressão e superioridade geral a partir de uma única estatística.

A soma de xG descreve as oportunidades registradas segundo o modelo. Ela não determina quem deveria vencer.

Associe cada frase a uma chave de evidência, permitindo conferir os números que a sustentam.

**16. Preparação para um modelo de linguagem leve**

Mantenha uma interface simples para geração de texto:

`generate_report(facts) -> report`

A primeira implementação usa regras. Uma implementação futura poderá usar um modelo pequeno local.

Prepare o armazenamento separado de:

- Fatos de entrada.
- Versão das métricas.
- Texto gerado.
- Correção humana aprovada, quando houver.
- Identificação da partida e do recorte.

Isso permitirá construir, mais adiante, exemplos de “fatos estruturados → relatório revisado”.

O futuro modelo de linguagem deve redigir a partir dos fatos calculados. A aritmética continuará na camada de métricas.

Não faça fine-tuning nesta fase. Primeiro precisamos de análises corretas e exemplos revisados.

**17. Etapas de implementação**

**Etapa 1 — Uma partida e um mapa de chutes**

Entregar:

- Estrutura mínima do projeto.
- Dependências declaradas.
- Seleção e carregamento de uma partida real.
- Cache dos dados necessários.
- Interface com identificação da partida e mapa de chutes.
- Instruções para executar.
- Testes básicos de leitura, coordenadas e separação da disputa de pênaltis.

Critério de conclusão: conseguir abrir a aplicação, reconhecer a partida e visualizar as finalizações corretas. Depois de baixar os dados, repetir a análise usando o cache.

Nesta etapa, o tamanho dos marcadores pode ser uniforme. A integração de xG pertence à etapa seguinte.

**Etapa 2 — Análise descritiva com xG da fonte**

Entregar:

- Estatísticas por time.
- Mapas de passes, ações defensivas e calor.
- xG StatsBomb por chute e acumulado.
- Filtros de time e período.
- Resumo por regras.
- Exportação de gráficos e fatos estruturados.

Critério de conclusão: números, gráficos e texto concordam no mesmo recorte; todos os valores de xG estão identificados como StatsBomb.

**Etapa 3 — Base de treinamento reproduzível**

Entregar:

- Configuração do recorte.
- Download incremental e cache.
- Manifesto das partidas.
- Tabela de finalizações elegíveis.
- Atributos geométricos.
- Divisões de treino, validação e teste.
- Diagnóstico de cobertura, valores ausentes e exclusões.

Critério de conclusão: reproduzir as divisões e demonstrar que nenhuma partida aparece em mais de uma delas.

**Etapa 4 — Primeiro modelo próprio de npxG**

Entregar:

- Referência de taxa média.
- Pipeline de regressão logística.
- Avaliação na validação.
- Avaliação final no teste após escolher o modelo.
- Gráfico de calibração.
- Artefato salvo com metadados.

Critério de conclusão: previsões válidas entre zero e um, execução reproduzível e comparação documentada com a referência.

**Etapa 5 — Modelo próprio no painel**

Entregar:

- Inferência com o modelo salvo.
- Mapas e curva acumulada com npxG próprio.
- Identificação da origem e versão.
- Componente de pênaltis para completar o xG próprio.
- Comparação visual com o xG da StatsBomb.
- Texto atualizado conforme a origem selecionada.

Critério de conclusão: cada total usa a fonte escolhida de forma consistente e o painel funciona com o modelo previamente treinado.

**Etapa 6 — Consolidação**

Entregar:

- Tratamento claro de falhas de download e dados ausentes.
- Exportação de fatos em JSON, estatísticas em CSV, gráficos em PNG e resumo em Markdown.
- Documentação de instalação, métricas e limitações.
- Verificação de uma partida com prorrogação e disputa de pênaltis.
- Documentação da interface para o futuro gerador de texto leve.

Critério de conclusão: uma pessoa consegue instalar, executar a demonstração e compreender a origem dos números.

**18. Testes que realmente importam**

Priorize testes para os riscos do projeto:

- Duplicação de eventos.
- Coordenadas e orientação do campo.
- Cálculo da distância e do ângulo.
- Separação de partidas entre treino, validação e teste.
- Ausência de atributos com informação posterior ao chute.
- Exclusão da disputa de pênaltis.
- Reconciliação de placar com gols contra.
- Passes completos conforme a semântica da StatsBomb.
- Casos sem chutes ou sem passes.
- Totais de xG consistentes com a tabela exibida.
- Concordância entre fatos e resumo.
- Uso do cache sem rede.
- Carregamento do modelo salvo.

Os testes automatizados principais devem usar pequenas amostras locais ou dados sintéticos. A integração com a fonte real pode ser verificada separadamente.

**19. Como conduzir o trabalho**

Antes de editar, examine o projeto existente e suas instruções locais. Preserve trabalho já realizado.

Implemente apenas a Etapa 1 nesta execução.

Ao terminar, explique:

- O que foi entregue.
- Como executar.
- Qual partida foi usada.
- Quais verificações passaram.
- Quais limitações existem.
- Qual será a próxima entrega.

Tome decisões rotineiras de implementação com base nesta especificação. Pergunte quando faltar uma informação que realmente impeça a etapa atual.

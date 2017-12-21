# Perceptron Multicamadas

Implementação paralela do Perceptron Multicamadas em OpenACC,
com o objetivo de comparar a performance e eficiência entre outras
implementações do Perceptron Multicamadas (CUDA e sequencial), já
que o treinamento da rede é uma operação altamente custosa.

## A implementação

Como o foco do implementação não estava na eficiência do treinamento
da rede, e sim no na comparação de perfromance para meu TCC, a
implementação utiliza um modelo para o treinamento da rede neural bem
simples, utilizando o algoritmo _backpropagation_, já que a rede pode
possuir mais de uma camada.

## Como compilar

Para compilar, são necessários o utilitário __GNU make__ para
automatizar o processo de compilação, e um compilador que suporte a
especificação OpenACC, como o __PGI Compiler__ (o arquivo Makefile foi
escrito especificamente para este compilador, já que o mesmo é
gratuito).

Lembrando que, apesar de um dos objetivos da especificação OpenACC é
trazer portabilidade, essa implementação utiliza rotinas da API CUDA
para transferência dos dados entre a CPU e GPU (já que nessa
implementação, o foco não está na portabilidade).

```sh
make # Executar este comando na pasta raíz do projeto (e torcer para compilar!)
```

## As etapas (básicas)  para o treinamento da rede

Há uma estrutura específica para os armazenar os padrões de
treinamento, onde a mesma consiste basicamente de dois vetores, um
vetor com a amostra à ser apresentada à rede e outra com a saída
desejada para rede:

```c
/**
 * Estrutura que irá representar um padrão de treinamento
 * para ser apresentado à rede.
 */
typedef struct {
  /** A vetor com a amostra (entrada da rede). */
  const float * d_amostra;
  
  /** Vetor com os valores desejados para saída da rede (objetivo). */
  const float * d_alvo;

} PadraoTreinamento;	
```

Essa estrutura, pode ser populada tanto manualmente, ou através de
dois arquivos no formato CSV (um com as amostras e outro com os
vetores de objetivo), utilizando a função abaixo (lembrando que os
dados serão carregados na memória do dispositivo acelerador):

```c
/**
 * Método que carrega os padrões de treinamento de dois arquivos para a memória do
 * dispositivo acelerador, um com as amostras (que são normalizadas pela função), 
 * onde cada linha do mesmo representa uma amostra (com os valores separados
 * por ponto e vírgula ";"), e outro com o vetor de objetivos para cada
 * amostra respectivamente (com os valores separados por ponto e vírgula também).
 *
 * @param nomeArquivoAmostras Nome do arquivo (com extensão) com as amostras
 *                            dos padrões de treinamento ou de teste.
 *
 * @param nomeArquivoAlvos Nome do arquivo (com extensão) com os
 *                         objetivos dos padrões de treinamento ou
 *                         de teste.
 *
 * @param menorValAmostra Menor valor presente nas amostras (para normalização).
 *
 * @param maiorValAmostra Maior valor presente nas amostras (para normalização).
 *
 * @param qtdItensAmostra Quantidade de itens por amostra.
 *
 * @param qtdItensAlvo Quantidade de itens por vetor de objetivo.
 *
 * @param qtdPadroes Quantidade de padrões nos arquivos.
 *
 * @return Vetor com os padrões de treinamento ou de teste carregados ou
 *         NULO caso não seja possível abrir os arquivos para leitura.
 */
PadraoTreinamento *
PadraoTreinamento_carregarPadroesArquivo(char * nomeArquivoAmostras,
                                         char * nomeArquivoAlvos,
                                         float menorValAmostra,
                                         float maiorValAmostra,
                                         int qtdItensAmostra,
                                         int qtdItensAlvo,
                                         int qtdPadroes);
```

Também é necessário alocar a estrutura do Perceptron Multicamadas na
memória utilizando a seguinte função:

```c
/**
 * Método que aloca o Perceptron Multicamadas na memória do hospedeiro,
 * salvo os atributos das camadas que serão salvos no dispositivo 
 * acelerador.
 *
 * @param qtdNeuroniosEntrada Quantidade de neurônios da camada de
 *                              entrada.
 *
 * @param qtdCamadas Quantidade de camadas (em que há processamento).
 *
 * @param qtdNeuroniosCamada Vetor com a quantidade de neurônios para cada
 *                           camada.
 *
 * @param funcaoAtivacaoRede Função de ativação da rede, ou seja, todas as
 *                           camadas da rede estarão atribuidas para serem
 *                           ativadas com tal função (usar a enumeração
 *                           "FuncoesAtivacaoEnum"). Lembrando que, caso se
 *                           deseje que as camadas tenham funções de ativação
 *                           diferente, estabelecer manualmente estes valores
 *                           nas respectivas camadas.
 *
 * @return Referência para a estrutura alocada.
 */
PerceptronMulticamadas *
PerceptronMulticamadas_inicializar(int qtdNeuroniosEntrada,
                                   int qtdCamadas,
                                   int * qtdNeuroniosCamada,
                                   int funcaoAtivacaoRede);
```

FINALMENTE, agora é a hora de iniciar o treinamento com a função abaixo:

```c
/**
 * Método que realiza o "backpropagation" da rede de forma paralela no
 * dispositivo acelerador através dos padrões de treinamento até que o 
 * erro da rede seja menor ou igual ao erro desejado OU o treinamento atinga a
 * quantidade máxima de epocas (QTD_MAX_EPOCAS).
 *
 * @param pm Perceptron.
 *
 * @param padroes Padrões para treinamento.
 *
 * @param qtdPadroesTreinamento Quantidade de padrões de treinamento.
 *
 * @param taxaAprendizagem Taxa de aprendizagem.
 *
 * @param erroDesejado Condição de parada para o treinamento
 *                     da rede.
 *
 * @param gerarHistorico Se será necessário gerar o histórico ou não.
 *
 * @return Histórico do treinamento. 
 */
HistoricoTreinamento *
PerceptronMulticamadas_backpropagation(PerceptronMulticamadas * pm,
				       PadraoTreinamento * padroes,
				       int qtdPadroesTreinamento,
				       float taxaAprendizagem,
				       float erroDesejado,
				       bool gerarHistorico);
```

Para demais informações em relação as funções, basta olhar os arquivos
de cabeçalho, pois as funções estão devidamente documentadas (acredito
eu).

// Esse include abaixo já possui o include do cabeçalho
// "perceptron_multicamadas.h" 
#include "historico_treinamento.h"

PerceptronMulticamadas *
PerceptronMulticamadas_inicializar(int qtdNeuroniosEntrada,
                                   int qtdCamadas,
                                   int * qtdNeuroniosCamada,
                                   int funcaoAtivacaoRede)
{
  /* Criando o vetor de referências que irá armazenar as camadas. */
  Camada ** camadas = malloc(sizeof(Camada *) * qtdCamadas);

  /* Alocando as camadas. */
  camadas[0] = __alocarCamada(qtdNeuroniosCamada[0], qtdNeuroniosEntrada,
                              funcaoAtivacaoRede);

  for (int i = 1; i < qtdCamadas; i++)
  {
    camadas[i] = __alocarCamada(qtdNeuroniosCamada[i],qtdNeuroniosCamada[i - 1],
                                funcaoAtivacaoRede);
  }

  /* Alocando a estrutura que irá abrigar as camadas. */
  PerceptronMulticamadas * pm = malloc(sizeof(PerceptronMulticamadas));

  /* Preenchendo os atributos do Perceptron. */
  pm->camadas = camadas;
  pm->qtdCamadas = qtdCamadas;
  pm->qtdNeuroniosEntrada = qtdNeuroniosEntrada;

  /* Retornando a estrutura alocada. */
  return pm;
}

Camada * __alocarCamada(int qtdNeuronios,
			int qtdPesosNeuronio,
                        int funcaoAtivacao)
{
  /* Alocando a camada. */
  Camada * camada = malloc(sizeof(Camada));

  /* Gerando o vetor de pesos aleatórios dos neurônios no
     hospedeiro. */
  float * h_vetorPesos = __alocarVetorPesosRandomicos(qtdPesosNeuronio
						      *	qtdNeuronios);

  /* Copiando o vetor de pesos aleatórios gerados no hospedeiro para
     o dispositivo acelerador. */
  cudaMalloc((void **) &camada->d_W, sizeof(float) *
	     (qtdPesosNeuronio * qtdNeuronios));
  cudaMemcpy(camada->d_W, h_vetorPesos, sizeof(float) *
	     (qtdPesosNeuronio * qtdNeuronios), cudaMemcpyHostToDevice);
  
  /* Alocando o vetor que irá armazenar a ativação dos neurônios
     no dispositivo acelerador. */
  cudaMalloc((void **) &camada->d_neuronioAtivacao, sizeof(float) * qtdNeuronios);

  /* Alocando o vetor que irá armazenar as derivadas dos neurônios no
     dispositivo acelerador. */
  cudaMalloc((void **) &camada->d_neuronioDerivada, sizeof(float) * qtdNeuronios);

  /* Alocando o vetor que irá armazenar o erro retropropagado calculado para
     cada neurônio no dispositivo acelerador. */
  cudaMalloc((void **) &camada->d_neuronioErroRprop, sizeof(float) * qtdNeuronios);

  /* Alocando vetor de bias no hospedeiro. */
  float * h_bias = malloc(sizeof(float) * qtdNeuronios);

  /* Inicializando os valores do vetor de bias através da macro
     definida no arquivo de cabeçalho (BIAS). */
  for (int i = 0; i < qtdNeuronios; i++)
  {
    h_bias[i] = BIAS;
  }

  /* Desalocando os vetores do hospedeiro que já foram copiados
     para o dispositivo acelerador. */
  free(h_vetorPesos);
  free(h_bias);

  /* Alocando o espaço para o vetor de bias no adaptador gráfico
   * e copiando o vetor de bias do hospedeiro para o mesmo.
   */
  cudaMalloc((void **) &camada->d_bias, sizeof(float) * qtdNeuronios);
  cudaMemcpy(camada->d_bias, h_bias, sizeof(float) * qtdNeuronios,
	     cudaMemcpyHostToDevice);
  
  /* Preenchendo os demais atributos. */
  camada->qtdNeuronios = qtdNeuronios;
  camada->funcaoAtivacao = funcaoAtivacao;

  /* Retornando a referência para a camada alocada. */
  return camada;
}

float * __alocarVetorPesosRandomicos(int qtdPesos)
{
  /* Alocando o vetor de pesos. */
  float * vetorPesos;
  vetorPesos = malloc(sizeof(float) * qtdPesos);

  /* Gerando a semente (número entre 0 a 99 999)... */
  srand(time(NULL));
  int semente = rand() % 100000;

  for (int i = 0; i < qtdPesos; i++)
  {
    /* Gerando o peso no intervalo de RAND_LIM_MIN..RAND_LIM_MAX. */
    vetorPesos[i] = r4_uniform_ab(RAND_LIM_MIN, RAND_LIM_MAX, &semente);
  }

  /* Retornando a referência para o vetor alocado. */
  return vetorPesos;
}

void Camada_calcularAtivacaoNeuroniosPrimeiraCamada(const Camada camada,
                                                    const float * d_amostra,
                                                    int qtdNeuroniosEntrada)
{
  /* Convertendo a estrutura da camada para variáveis de tipos
   * primitivos para que o OpenACC não tente copiar os vetores
   * para a memória do dispositivo, pois os mesmos já estão
   * na memória do dispositivo acelerador. */
  float * camada_d_W = camada.d_W;
  float * camada_d_neuronioAtivacao = camada.d_neuronioAtivacao;
  float * camada_d_neuronioDerivada = camada.d_neuronioDerivada;
  float * camada_d_neuronioErroRprop = camada.d_neuronioErroRprop;
  float * camada_d_bias = camada.d_bias;
  
  /* Percorrendo todos os neurônios da camada de forma paralela
   * no dispositivo acelerador. */
  #pragma acc parallel loop vector_length(TAM_VECTOR) gang, vector \
  copyin(camada, qtdNeuroniosEntrada) \
  deviceptr(camada_d_W, camada_d_neuronioAtivacao, \
            camada_d_neuronioDerivada, camada_d_neuronioErroRprop, \
	    camada_d_bias, d_amostra)
  for (int n = 0; n < camada.qtdNeuronios; n++)
  {  
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada_d_W[qtdNeuroniosEntrada * n];
    
    /* Calculando o valor da função de integração o neurônio. */
    float valFuncIntegracao = 0.0;

    #pragma acc loop seq reduction(+:valFuncIntegracao)
    for (int i = 0; i < qtdNeuroniosEntrada; i++)
    {      
      /* Somando o item "i-ésimo" da amostra pelo peso "i-ésimo" do
      neurônio "n-ésimo". */
      valFuncIntegracao += w[i] * d_amostra[i];
    }
    
    /* Por fim calculando a ativação do neurônio (usando o bias) junto com
    sua derivada. */
    float ativacaoNeuronio;

    switch (camada.funcaoAtivacao)
    {
    case Identidade:
       camada_d_neuronioAtivacao[n] = valFuncIntegracao + camada_d_bias[n];
       camada_d_neuronioDerivada[n] = 1;
       break;
    case Degrau:
      ativacaoNeuronio = funcaoDegrau(valFuncIntegracao + camada_d_bias[n]);
      camada_d_neuronioAtivacao[n] = ativacaoNeuronio;
      camada_d_neuronioDerivada[n] = derivadaFuncaoDegrau(ativacaoNeuronio);
      break;
    case Sigmoide:
      ativacaoNeuronio = funcaoSigmoide(valFuncIntegracao + camada_d_bias[n]);
      camada_d_neuronioAtivacao[n] = ativacaoNeuronio;
      camada_d_neuronioDerivada[n] = derivadaFuncaoSigmoide(ativacaoNeuronio);
      break;
    case TangHiperbolica:
      ativacaoNeuronio = funcaoTangHiperbolica(valFuncIntegracao + camada_d_bias[n]);
      camada_d_neuronioAtivacao[n] = ativacaoNeuronio;
      camada_d_neuronioDerivada[n] = derivadaFuncaoTangHiperbolica(ativacaoNeuronio);
    }
  }
}

void Camada_calcularAtivacaoNeuroniosCamada(const Camada camadaAnterior,
                                            const Camada camada)
{
  /* Convertendo a estrutura da camada para variáveis de tipos
   * primitivos para que o OpenACC não tente copiar os vetores
   * para a memória do dispositivo, pois os mesmos já estão
   * na memória do dispositivo acelerador. */
  float * camadaAnterior_d_W = camadaAnterior.d_W;
  float * camadaAnterior_d_neuronioAtivacao = camadaAnterior.d_neuronioAtivacao;
  float * camadaAnterior_d_neuronioDerivada = camadaAnterior.d_neuronioDerivada;
  float * camadaAnterior_d_neuronioErroRprop = camadaAnterior.d_neuronioErroRprop;
  float * camadaAnterior_d_bias = camadaAnterior.d_bias;

  float * camada_d_W = camada.d_W;
  float * camada_d_neuronioAtivacao = camada.d_neuronioAtivacao;
  float * camada_d_neuronioDerivada = camada.d_neuronioDerivada;
  float * camada_d_bias = camada.d_bias;

  /* Percorrendo todos os neurônios da camada de forma paralela
     no dispositivo acelerador. */
  #pragma acc parallel loop vector_length(TAM_VECTOR) gang, vector \
  deviceptr(camadaAnterior_d_W, camadaAnterior_d_neuronioAtivacao, \
            camadaAnterior_d_neuronioDerivada, camadaAnterior_d_bias, \
            camadaAnterior_d_neuronioErroRprop, \
            camada_d_neuronioAtivacao, camada_d_neuronioDerivada, \
	    camada_d_bias)
  for (int n = 0; n < camada.qtdNeuronios; n++)
  {
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada_d_W[camadaAnterior.qtdNeuronios * n];

    /* Calculando o valor da função de integração o neurônio. */
    float valFuncIntegracao = 0.0;

    #pragma acc loop seq reduction(+:valFuncIntegracao)
    for (int i = 0; i < camadaAnterior.qtdNeuronios; i++)
    {
      /* Somando a ativação do neurônio "i-ésimo" pelo peso "i-ésimo" do
      neurônio "n-ésimo". */
      valFuncIntegracao += w[i] * camadaAnterior_d_neuronioAtivacao[i];
    }
    
    /* Por fim calculando a ativação do neurônio (usando o bias) junto com
    sua derivada. */
    float ativacaoNeuronio;

    switch (camada.funcaoAtivacao)
    {
    case Identidade:
       camada_d_neuronioAtivacao[n] = valFuncIntegracao + camada_d_bias[n];
       camada_d_neuronioDerivada[n] = 1;
       break;
    case Degrau:
      ativacaoNeuronio = funcaoDegrau(valFuncIntegracao + camada_d_bias[n]);
      camada_d_neuronioAtivacao[n] = ativacaoNeuronio;
      camada_d_neuronioDerivada[n] = derivadaFuncaoDegrau(ativacaoNeuronio);
      break;
    case Sigmoide:
      ativacaoNeuronio = funcaoSigmoide(valFuncIntegracao + camada_d_bias[n]);
      camada_d_neuronioAtivacao[n] = ativacaoNeuronio;
      camada_d_neuronioDerivada[n] = derivadaFuncaoSigmoide(ativacaoNeuronio);
      break;
    case TangHiperbolica:
      ativacaoNeuronio = funcaoTangHiperbolica(valFuncIntegracao + camada_d_bias[n]);
      camada_d_neuronioAtivacao[n] = ativacaoNeuronio;
      camada_d_neuronioDerivada[n] = derivadaFuncaoTangHiperbolica(ativacaoNeuronio);
    }
  }
}

void Camada_calcularErroRpropNeuroniosCamada(const Camada camada,
                                             const Camada camadaPosterior)
{
  /* Convertendo a estrutura da camada para variáveis de tipos
   * primitivos para que o OpenACC não tente copiar os vetores
   * para a memória do dispositivo, pois os mesmos já estão
   * na memória do dispositivo acelerador. */
  float * camada_d_neuronioDerivada = camada.d_neuronioDerivada;
  float * camada_d_neuronioErroRprop = camada.d_neuronioErroRprop;

  float * camadaPosterior_d_W = camadaPosterior.d_W;
  float * camadaPosterior_d_neuronioErroRprop = camadaPosterior.d_neuronioErroRprop;
  float * camadaPosterior_d_bias = camadaPosterior.d_bias;
  
  /* Percorrendo todos os neurônios da camada de forma paralela
     no dispositivo acelerador. */
  #pragma acc parallel loop vector_length(TAM_VECTOR) gang, vector \
  copyin(camada, camadaPosterior) \
  deviceptr(camada_d_neuronioDerivada, camada_d_neuronioErroRprop, \
            camadaPosterior_d_W, camadaPosterior_d_neuronioErroRprop, \
            camadaPosterior_d_bias)
  for (int n = 0; n < camada.qtdNeuronios; n++)
  {
    /* Calculando a soma dos erros da camada posterior multiplicados
    pelo respectivos pesos. */
    float somaErroCamadaPosterior = 0.0;

    #pragma acc loop seq reduction(+:somaErroCamadaPosterior)
    for (int i = 0; i < camadaPosterior.qtdNeuronios; i++)
    {
      /* Coletando o peso do neurônio "i-ésimo" da camada posterior
      que se conecta ao respectivo neurônio "n-ésimo" que está tendo seu
      erro calculado. */
      float w = camadaPosterior_d_W[camada.qtdNeuronios * i + n];

      /* Calculando o erro do neurônio "i-ésimo" da camada posterior
      multiplicado pelo respectivo peso da camada posterior que se
      conecta ao respectivo neurônio "n-ésimo" que está tendo seu erro
      calculado, e somando... */
      somaErroCamadaPosterior += w * camadaPosterior_d_neuronioErroRprop[i];
    }

    /* Por fim, calculando o erro retropropagado do neurônio. */
    camada_d_neuronioErroRprop[n] = camada_d_neuronioDerivada[n] *
      somaErroCamadaPosterior;
  }
}

void Camada_calcularErroRpropNeuroniosUltimaCamada(const Camada camada,
						   const float * d_alvo,
						   float * d_erroPadrao)
{
  /* Convertendo a estrutura da camada para variáveis de tipos
   * primitivos para que o OpenACC não tente copiar os vetores
   * para a memória do dispositivo, pois os mesmos já estão
   * na memória do dispositivo acelerador. */
  float * camada_d_neuronioAtivacao = camada.d_neuronioAtivacao;
  float * camada_d_neuronioDerivada = camada.d_neuronioDerivada;
  float * camada_d_neuronioErroRprop = camada.d_neuronioErroRprop;
  float * camada_d_bias = camada.d_bias;
  
  /* Calculando o erro retropropagado da última camada no próprio dispositivo
     acelerador de forma sequencial para manter a localidade dos dados. */
  #pragma acc parallel copyin(camada) \
  deviceptr(camada_d_neuronioAtivacao, camada_d_neuronioDerivada, \
            camada_d_neuronioErroRprop, camada_d_bias, \
            d_alvo, d_erroPadrao)
  {
    /* Variável que irá armazenar o erro para o padrão apresentado à rede. */
    float erroPadrao = 0.0;

    /* Percorrendo todos os neurônios da camada de forma SEQUENCIAL. */
    #pragma acc loop seq reduction(+:erroPadrao)
    for (int n = 0; n < camada.qtdNeuronios; n++)
    {
      /* Calculando o erro da saída do neurônio "n-ésimo". */
      float erroSaidaNeuronio = camada_d_neuronioAtivacao[n] - d_alvo[n];
	
      /* Calculando o erro retropropagado. */
      camada_d_neuronioErroRprop[n] = erroSaidaNeuronio *
	camada_d_neuronioDerivada[n];

      /* Calculando o erro para o padrão... */
      erroPadrao += 0.5 * powf(erroSaidaNeuronio, 2);
    }
    
    /* "Retornando" o erro do padrão calculado para esta amostra. */
    *d_erroPadrao = erroPadrao;
  }
}

void Camada_atualizarPesosNeuroniosPrimeiraCamada(const Camada camada,
                                                  const float * d_amostra,
                                                  int qtdNeuroniosEntrada,
                                                  float taxaAprendizagem)
{
  /* Convertendo a estrutura da camada para variáveis de tipos
   * primitivos para que o OpenACC não tente copiar os vetores
   * para a memória do dispositivo, pois os mesmos já estão
   * na memória do dispositivo acelerador. */
  float * camada_d_W = camada.d_W;
  float * camada_d_neuronioErroRprop = camada.d_neuronioErroRprop;
  float * camada_d_bias = camada.d_bias;
  
  /* Percorrendo todos os neurônios da camada de forma paralela
     no dispositivo acelerador. */
  #pragma acc parallel loop vector_length(TAM_VECTOR) gang, vector \
  copyin(camada, qtdNeuroniosEntrada, taxaAprendizagem) \
  deviceptr(camada_d_W, camada_d_neuronioErroRprop, \
            camada_d_bias, d_amostra)
  for (int n = 0; n < camada.qtdNeuronios; n++)
  {
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada_d_W[qtdNeuroniosEntrada * n];

    /* Percorrendo todos os pesos do neurônio. */
    #pragma acc loop seq
    for (int i = 0; i < qtdNeuroniosEntrada; i++)
    {
      /* Atualizando o peso "i-ésimo" do neurônio "n-ésimo". */
      w[i] += -taxaAprendizagem * d_amostra[i] * camada_d_neuronioErroRprop[n];
    }

    /* Atualizando o bias do neurônio "n-ésimo"... */
    camada_d_bias[n] += -taxaAprendizagem * camada_d_neuronioErroRprop[n];
  }
}

void Camada_atualizarPesosNeuroniosCamada(const Camada camadaAnterior,
                                          const Camada camada,
                                          float taxaAprendizagem)
{
  /* Convertendo a estrutura da camada para variáveis de tipos
   * primitivos para que o OpenACC não tente copiar os vetores
   * para a memória do dispositivo, pois os mesmos já estão
   * na memória do dispositivo acelerador. */
  float * camadaAnterior_d_W = camadaAnterior.d_W;
  float * camadaAnterior_d_neuronioAtivacao = camadaAnterior.d_neuronioAtivacao;
  float * camadaAnterior_d_neuronioDerivada = camadaAnterior.d_neuronioDerivada;
  float * camadaAnterior_d_neuronioErroRprop = camadaAnterior.d_neuronioErroRprop;
  float * camadaAnterior_d_bias = camadaAnterior.d_bias;
  
  float * camada_d_W = camada.d_W;
  float * camada_d_neuronioAtivacao = camada.d_neuronioAtivacao;
  float * camada_d_neuronioDerivada = camada.d_neuronioDerivada;
  float * camada_d_neuronioErroRprop = camada.d_neuronioErroRprop;
  float * camada_d_bias = camada.d_bias;
  
  /* Percorrendo todos os neurônios da camada de forma paralela
     no dispositivo acelerador. */
  #pragma acc parallel loop vector_length(TAM_VECTOR) gang, vector \
  copyin(camadaAnterior, camada, taxaAprendizagem) \
  deviceptr(camadaAnterior_d_neuronioAtivacao,	\
            camada_d_W, camada_d_neuronioAtivacao, \
            camada_d_neuronioErroRprop, \
            camada_d_bias)
  for (int n = 0; n < camada.qtdNeuronios; n++)
  {
    /* Variável que irá referênciar os pesos do neurônio "n-ésimo".
    Os pesos serão obtidos utilizando o deslocamento "row-major". */
    float * w = &camada_d_W[camadaAnterior.qtdNeuronios * n];

    /* Percorrendo todos os pesos do neurônio. */
    #pragma acc loop seq
    for (int i = 0; i < camadaAnterior.qtdNeuronios; i++)
    {
      /* Atualizando o peso "i-ésimo" do neurônio "n-ésimo". */
      w[i] += -taxaAprendizagem * camadaAnterior_d_neuronioAtivacao[i] *
              camada_d_neuronioErroRprop[n];
    }

    /* Atualizando o bias do neurônio "n-ésimo"... */
    camada_d_bias[n] += -taxaAprendizagem * camada_d_neuronioErroRprop[n];
  }
}

void PerceptronMulticamadas_feedfoward(PerceptronMulticamadas * pm,
                                       const float * d_amostra)
{
  /* Calculando a ativação dos neurônios da primeira camada. */
  Camada_calcularAtivacaoNeuroniosPrimeiraCamada(*pm->camadas[0], d_amostra,
                                                 pm->qtdNeuroniosEntrada);

  /* Calculando a ativação dos neurônios das demais camadas. */
  for (int c = 1; c < pm->qtdCamadas; c++)
  {
    Camada_calcularAtivacaoNeuroniosCamada(*pm->camadas[c - 1],
					   *pm->camadas[c]);
  }
}

HistoricoTreinamento *
PerceptronMulticamadas_backpropagation(PerceptronMulticamadas * pm,
				       PadraoTreinamento * padroes,
				       int qtdPadroesTreinamento,
				       float taxaAprendizagem,
				       float erroDesejado,
				       bool gerarHistorico)
{
  /* Inicializando a estrutura. */
  HistoricoTreinamento * historicoTreinamento; 
  if (gerarHistorico)
  {
    historicoTreinamento = HistoricoTreinamento_inicializar(pm,
							    taxaAprendizagem,
							    erroDesejado);
  }
  
  /* Variável que irá armazenar o erro global calculado para cada iteração.
     A mesma será populada após a execução de uma iteração, onde será copiado
     o erro global que está na memória do dispositivo acelerador para o
     hospedeiro. */
  float h_erroGlobal;

  /* Alocando na memória do dispositivo acelerador  a variável que 
     irá armazenar o erro para os padrões apresentados à rede. */
  float * d_erroPadrao;
  cudaMalloc((void **) &d_erroPadrao, sizeof(float));

  /* O treinamento irá ocorrer enquanto o erro da rede estiver acima
     do desejado OU a quantidade de épocas não tenha atingido o limite. */
  int epocas = 0;

  do
  {
    /* Inicializando com 0 o erro global no para
       a época atual. */
    h_erroGlobal = 0;

    /* Variáveis que serão utilizadas para armazenar a hora que foi iniciada
       e finalizada o treinamento da rede para a época atual. */
    struct timeval horaAntesTreinamento;
    struct timeval horaDepoisTreinamento;

    /* Coletando a hora antes do treinamento. */
    gettimeofday(&horaAntesTreinamento, NULL);

    /* Apresentando os padrões de treinamento para rede e realizando o
       treinamento da mesma. */
    for (int i = 0; i < qtdPadroesTreinamento; i++)
    {
      
      /* Alimentando a rede com o padrão "i-ésimo". */
      PerceptronMulticamadas_feedfoward(pm, padroes[i].d_amostra);

      /* Calculando o erro dos neurônios da última camada e já calculando
	 o erro global para o padrão apresentado à rede. */
      Camada_calcularErroRpropNeuroniosUltimaCamada
	(*pm->camadas[pm->qtdCamadas - 1], padroes[i].d_alvo, d_erroPadrao);

      /* Copiando o erro do padrão armazenado no adaptador gráfico para o hospedeiro. */
      float h_erroPadrao;
      cudaMemcpy(&h_erroPadrao, d_erroPadrao, sizeof(float),
		 cudaMemcpyDeviceToHost);
      
      /* Somando o erro calculado para o padrão no erro global. */
      h_erroGlobal += h_erroPadrao;
       
      /* Realizando a retropropagação do erro para as demais camadas. */
      for (int c = pm->qtdCamadas - 2; c >= 0; c--)
      {
        Camada_calcularErroRpropNeuroniosCamada(*pm->camadas[c],
                                                *pm->camadas[c + 1]);
      }

      /* Atualizando os pesos dos neurônios da primeira camada. */
      Camada_atualizarPesosNeuroniosPrimeiraCamada(*pm->camadas[0],
                                                   padroes[i].d_amostra,
                                                   pm->qtdNeuroniosEntrada,
                                                   taxaAprendizagem);

      /* Atualizando os pesos dos neurônios das demais camadas. */
      for (int c = 1; c < pm->qtdCamadas; c++)
      {
        Camada_atualizarPesosNeuroniosCamada(*pm->camadas[c - 1],
                                             *pm->camadas[c],
                                             taxaAprendizagem);
      }
    }

    /* Coletando a hora depois do treinamento. */
    gettimeofday(&horaDepoisTreinamento, NULL);

    /* Realizando o cálculo do MSE. */
    h_erroGlobal = h_erroGlobal/ qtdPadroesTreinamento;

    /* Atualizando a quantidade de épocas. */
    epocas++;

    /* Calculando o tempo de treinamento. */
    float segs =  (horaDepoisTreinamento.tv_sec +
		   horaDepoisTreinamento.tv_usec / 1000000.0) -
                  (horaAntesTreinamento.tv_sec +
                   horaAntesTreinamento.tv_usec / 1000000.0);

    /* Adicionando as informações desta época no histórico de
    treinamento. */
    if (gerarHistorico)
    {
      HistoricoTreinamento_adicionarInfoEpoca(historicoTreinamento,
					      segs,
					      h_erroGlobal);
    }
 
    
     /* Mostrando a época e o erro MSE para a mesma (caso tenha que ser feito). */
    if (INFO_ESTATISTICAS)
    {
      printf("Época: %d\nErro MSE: %.4f\n", epocas, h_erroGlobal);
      printf("Tempo total de execução da época: %.2f segundo(s)\n\n", segs);
    }
    
  } while (h_erroGlobal > erroDesejado && epocas < QTD_MAX_EPOCAS);

  /* Desalocando as variáveis que estão no dispositivo acelerador 
     que não serão mais necessárias. */
  cudaFree(d_erroPadrao);
  
  return historicoTreinamento;
}

void embaralhamentoFisherYates(int * v, int n)
{
  /* Gerando a semente para gerar
  os números randômicos. */
  srand(time(NULL));

  /* Percorrendo o vetor do fim para o
  início. */
  for (int i = n - 1; i > 0; i--)
  {
    /* Gerando o índice para permutação
    entre o número "i-ésimo" e algum
    número entre 0 e i-1. */
    int indicePerm = rand() % i;

    /* Realizando a troca. */
    int auxTroca = v[i];
    v[i] = v[indicePerm];
    v[indicePerm] = auxTroca;
  }
}

void normalizacaoMinMax(float * v, int n, float min, float max)
{
  /* Percorrendo todos os itens do vetor e realizando a normalização
  dos mesmos. */
  for (int i = 0; i < n; i++)
  {
    v[i] = (v[i] - min) / (max - min);
  }
}

#pragma acc routine seq
inline float funcaoDegrau(float z)
{
  return (z >= 0) ? 1 : 0;
}

#pragma acc routine seq
float derivadaFuncaoDegrau(float valDegrau)
{
  return 1.0;
}

#pragma acc routine seq
inline float funcaoSigmoide(float z)
{
  return 1.0 / ((1.0) + expf(-z));
}

#pragma acc routine seq
inline float derivadaFuncaoSigmoide(float valSigmoide)
{
  return valSigmoide * (1.0 - valSigmoide);
}

#pragma acc routine seq
inline float funcaoTangHiperbolica(float z)
{
  return tanhf(z);
}

#pragma acc routine seq
inline float derivadaFuncaoTangHiperbolica(float valTangHiperbolica)
{
  return 1 - (valTangHiperbolica * valTangHiperbolica);
}

PadraoTreinamento *
PadraoTreinamento_carregarPadroesArquivo(char * nomeArquivoAmostras,
                                         char * nomeArquivoAlvos,
                                         float menorValAmostra,
                                         float maiorValAmostra,
                                         int qtdItensAmostra,
                                         int qtdItensAlvo,
                                         int qtdPadroes)
{
  /* Tentando abrir os arquivos para leitura. */
  FILE * arqAmostras;
  FILE * arqAlvos;

  arqAmostras = fopen(nomeArquivoAmostras, "r");
  arqAlvos = fopen(nomeArquivoAlvos, "r");

  /* Verificando se os arquivos foram abertos com sucesso. */
  if (arqAmostras == NULL || arqAlvos == NULL)
  {
    /* Retornando NULL. */
    return NULL;
  }

  /* Alocando o vetor que irá armazenar os padrões. */
  PadraoTreinamento * padroes;
  padroes = malloc(sizeof(PadraoTreinamento) * qtdPadroes);

  /* Primeiramente lendo as amostras e inserindo as mesmas nos respectivos
     padrões. */
  for (int i = 0; i < qtdPadroes; i++)
  {
    /* Coletando a linha com a amostra do arquivo. */
    char linhaAmostra[4096]; // 4 Kbytes
    fscanf(arqAmostras, "%s", linhaAmostra);

    /* Alocando o vetor para armazenar a amostra "i-ésima"
       no hospedeiro. */
    float * h_amostra = (float *) malloc(sizeof(float) * qtdItensAmostra);
    
    /* Extraindo o primeiro item da amostra. */
    h_amostra[0] = atof(strtok(linhaAmostra, ";\n\0"));

    /* Extraindo os demais itens da amostra. */
    for (int j = 1; j < qtdItensAmostra; j++)
    {
      /* Extraindo o item "j-ésimo" da amostra. */
      h_amostra[j] = atof(strtok(NULL, ";\n\0"));
    }

    /* Normalizando a amostra coletada... */
    normalizacaoMinMax(h_amostra, qtdItensAmostra, menorValAmostra,
                       maiorValAmostra);

    /* Copiando a amostra normalizada para a memória do dispositivo
       acelerador . */
    float * d_amostra;
    cudaMalloc((void **) &d_amostra, sizeof(float) * qtdItensAmostra);
    cudaMemcpy(d_amostra, h_amostra, sizeof(float) * qtdItensAmostra,
	       cudaMemcpyHostToDevice);
    
    /* Por fim, colocando no padrão a amostra acima extraida
    do arquivo. */
    padroes[i].d_amostra = d_amostra;

    /* Desalocando os vetores do hospedeiro que já foram copiados
       para o dispositivo acelerador. */
    free(h_amostra);
  }

  /* Lendo os vetores de alvo e inserindo os mesmos nos respectivos
     padrões. */
  for (int i = 0; i < qtdPadroes; i++)
  {
    /* Coletando a linha com o alvo do arquivo. */
    char linhaAlvo[4096]; // 4 Kbytes
    fscanf(arqAlvos, "%s", linhaAlvo);

    /* Alocando o vetor para armazenar o vetor de objetivo "i-ésimo"
       do hospedeiro. */
    float * h_alvo = (float *) malloc(sizeof(float) * qtdItensAlvo);

    /* Extraindo o primeiro item do vetor de objetivo. */
    h_alvo[0] = atof(strtok(linhaAlvo, ";\n\0"));

    /* Extraindo os demais itens do vetor de objetivo. */
    for (int j = 1; j < qtdItensAlvo; j++)
    {
      /* Extraindo o item "j-ésimo" do vetor de objetivo. */
      h_alvo[j] = atof(strtok(NULL, ";\n\0"));
    }

    /* Copiando o vetor objetivo para a memória do dispositivo 
       acelerador. */
    float * d_alvo;
    cudaMalloc((void **) &d_alvo, sizeof(float) * qtdItensAlvo);
    cudaMemcpy(d_alvo, h_alvo, sizeof(float) * qtdItensAlvo,
	       cudaMemcpyHostToDevice);
    
    /* Por fim, colocando no padrão o vetor de objetivo extraido do
       arquivo. */
    padroes[i].d_alvo = d_alvo;

    /* Desalocando os vetores do hospedeiro que já foram copiados
       para o dispositivo acelerador. */
    free(h_alvo);
  }

  fclose(arqAmostras);
  fclose(arqAlvos);

  return padroes;
}

float PerceptronMulticamadas_calcularTaxaAcerto(PerceptronMulticamadas * pm,
                                                PadraoTreinamento * padroesTeste,
                                                int qtdPadroesTeste)
{
  
  /* Alocando na memória do adaptador gráfico a variável que irá armazenar
     o erro calculado das iterações. */
  float * d_erroPadrao =  acc_malloc(sizeof(float));

  /* Percorrendo os padrões de teste. */
  float h_erroGlobal = 0;
  float h_erroPadrao;
  
  for (int i = 0; i < qtdPadroesTeste; i++)
  {
    /* Alimentando a rede com o padrão de teste "i-ésimo". */
    PerceptronMulticamadas_feedfoward(pm, padroesTeste[i].d_amostra);

    /* Calculando o erro dos neurônios da última camada. */
     Camada_calcularErroRpropNeuroniosUltimaCamada
       (*pm->camadas[pm->qtdCamadas - 1], padroesTeste[i].d_alvo,
	d_erroPadrao);

     /* Copiando o erro do padrão armazenado do dispositivo acelerador para
	o hospedeiro. */
     acc_memcpy_from_device(&h_erroPadrao, d_erroPadrao, sizeof(float));
     
     h_erroGlobal += h_erroPadrao;
  }

  /* Desalocando as variáveis do dispositivo acelerador que não
     serão mais utilizadas. */
  acc_free(d_erroPadrao);
  
  /* Retornando o erro MSE calculado. */
  return h_erroGlobal / qtdPadroesTeste;
}


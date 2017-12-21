#Compilação para execução paralela no adaptador gráfico
CXX=pgc++
CXXFLAGS=$(LIBS) -O3
CC=pgcc
CFLAGS=-O3 -Minform=inform -Minfo=all
LIBS=-cudalibs -acc -acclibs
TA=tesla:cc60,cuda8.0

prj_perceptron_multicamadas: main.o perceptron_multicamadas.o \
			     uniform.o historico_treinamento.o
	$(CXX) main.o perceptron_multicamadas.o historico_treinamento.o \
	uniform.o $(CXXFLAGS) -ta=$(TA) -o prj_perceptron_multicamadas

main.o: src/main.c
	$(CC) -c src/main.c $(CFLAGS) -ta=$(TA) -o main.o

perceptron_multicamadas.o: src/perceptron_multicamadas.c
	$(CC) -c src/perceptron_multicamadas.c $(CFLAGS) \
	-ta=$(TA) -o perceptron_multicamadas.o

uniform.o: src/uniform.c
	$(CC) -c src/uniform.c $(CFLAGS) \
	-o uniform.o

historico_treinamento.o: src/historico_treinamento.c
			 $(CC) -c src/historico_treinamento.c $(CFLAGS) \
			 -ta=$(TA) -o historico_treinamento.o

clean:
	rm *.o prj_perceptron_multicamadas

Uruchom 'python testLearnXOR.py', ażeby sprawdzić, czy pybrain jest u Ciebie ślicznie zainstalowany i możesz nauczać sieci neuronowe.
Uruchomienie 'python learn.py' uruchamia przykład.

Modyfikować należy jedynie moduł my_stuff:
 - należy zaimplementować klasę YourCoolMsiDataProvider
    * skombinować sobie własny dataset
    * wybrać rozmiar warstwy ukrytej sieci
    * konstruktor wczytuje dane dla sieci neuronowej
    * przykład przewiduje kurs USD względem PLN na podstawie kursów z poprzednich 5 dni
    * getTrainingData itp. powinno zwracać dataset w analogicznej postaci jak w przykładzie:
    ```
         [
            [ [kurs_wczoraj, kurs_przedwczoraj, ...], [kurs_dziś] ],
            [ [kurs_przedwczoraj, kurs_przedprzedwczoraj, ...], [kurs_wczoraj] ],
            ...
         ]
     ```

 - należy zaimplementować funkcję thingsYouDoWithTheNeuralNetwork
    * ma tu być uczenie sieci i testowanie dla serii 3 parametrów wymienionych w treści zadania
    * w przykładzie są obliczane wyniki dla wykresu błędu od ilości epok uczenia
    * podobnie jak w przykładzie można wypisywać wyniki - gotowe do importu w excelu i rysowania wykresu


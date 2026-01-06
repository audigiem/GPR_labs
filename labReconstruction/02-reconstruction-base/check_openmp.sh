#!/bin/bash
# Script de diagnostic pour OpenMP

echo "=========================================="
echo "Diagnostic OpenMP"
echo "=========================================="
echo ""

echo "1. Vérification du compilateur:"
echo "--------------------------------"
which gcc && gcc --version | head -1
which g++ && g++ --version | head -1
echo ""

echo "2. Test de compilation OpenMP:"
echo "--------------------------------"
cat > /tmp/test_openmp.cpp << 'EOF'
#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads disponibles: " << omp_get_max_threads() << std::endl;
    }
    return 0;
}
EOF

echo "Compilation du test..."
if g++ -fopenmp /tmp/test_openmp.cpp -o /tmp/test_openmp 2>&1; then
    echo "✓ Compilation réussie avec -fopenmp"
    echo "Exécution du test:"
    /tmp/test_openmp
    rm -f /tmp/test_openmp
else
    echo "✗ Échec de la compilation avec OpenMP"
fi
rm -f /tmp/test_openmp.cpp
echo ""

echo "3. Recherche de libgomp:"
echo "--------------------------------"
find /usr/lib* -name "libgomp*" 2>/dev/null | head -5
echo ""

echo "4. Vérification de l'exécutable actuel:"
echo "--------------------------------"
if [ -f "./02-reconstruction-base" ]; then
    echo "Bibliothèques liées:"
    ldd ./02-reconstruction-base | grep -i "gomp\|omp" || echo "Aucune bibliothèque OpenMP détectée"
    echo ""
    echo "Symboles OpenMP dans l'exécutable:"
    nm ./02-reconstruction-base 2>/dev/null | grep -i "omp_" | head -3 || echo "Aucun symbole OpenMP trouvé"
else
    echo "✗ Exécutable non trouvé"
fi
echo ""

echo "5. Vérification CMake:"
echo "--------------------------------"
if [ -f "CMakeCache.txt" ]; then
    echo "Variables OpenMP dans CMakeCache.txt:"
    grep -i "openmp" CMakeCache.txt | head -10 || echo "Aucune variable OpenMP trouvée"
else
    echo "✗ CMakeCache.txt non trouvé"
fi
echo ""

echo "=========================================="
echo "Fin du diagnostic"
echo "=========================================="


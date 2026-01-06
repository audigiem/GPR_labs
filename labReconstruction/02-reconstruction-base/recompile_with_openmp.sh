#!/bin/bash
# Script de recompilation avec OpenMP

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "Recompilation avec support OpenMP"
echo "=========================================="
echo ""

# Se placer dans le répertoire build
cd "$(dirname "$0")/build"
echo "Répertoire de travail: $(pwd)"
echo ""

# Nettoyer l'ancienne configuration
echo "1. Nettoyage de l'ancienne configuration..."
echo "--------------------------------------------"
rm -rf CMakeCache.txt CMakeFiles/
echo "✓ Cache CMake supprimé"
echo ""

# Reconfigurer avec CMake
echo "2. Configuration avec CMake..."
echo "--------------------------------------------"
cmake .. 2>&1 | tee cmake_output.log
echo ""

# Vérifier si OpenMP a été détecté
echo "3. Vérification de la détection OpenMP..."
echo "--------------------------------------------"
if grep -q "OpenMP found" cmake_output.log; then
    echo "✓ OpenMP détecté par CMake !"
    grep "OpenMP" cmake_output.log
else
    echo "⚠️  OpenMP non détecté - vérifier cmake_output.log"
    echo "Voici les messages CMake:"
    cat cmake_output.log
fi
echo ""

# Compiler
echo "4. Compilation du projet..."
echo "--------------------------------------------"
make clean 2>/dev/null || true
make -j$(nproc) 2>&1 | tee make_output.log
echo ""

# Vérifier l'exécutable
echo "5. Vérification de l'exécutable..."
echo "--------------------------------------------"
if [ -f "./02-reconstruction-base" ]; then
    echo "✓ Exécutable créé avec succès"
    echo ""
    echo "Taille: $(ls -lh ./02-reconstruction-base | awk '{print $5}')"
    echo ""
    echo "Bibliothèques OpenMP liées:"
    ldd ./02-reconstruction-base | grep -i "gomp\|omp" && echo "" || echo "⚠️  Aucune bibliothèque OpenMP détectée"
    echo ""
    echo "Macros de compilation utilisées:"
    grep -h "USE_PARALLEL_RBF\|_OPENMP" make_output.log | head -5 || echo "Pas de macros OpenMP trouvées dans la sortie de make"
else
    echo "✗ Échec de la compilation - exécutable non créé"
    echo "Vérifier make_output.log pour les erreurs"
    exit 1
fi
echo ""

echo "=========================================="
echo "Compilation terminée !"
echo "=========================================="
echo ""
echo "Pour lancer l'application:"
echo "  cd build"
echo "  ./02-reconstruction-base ../../points/frog.ply"
echo ""
echo "Dans le GUI, sélectionnez 'RBF' et vous devriez voir"
echo "l'option 'Use OpenMP Parallel RBF' apparaître."
echo ""


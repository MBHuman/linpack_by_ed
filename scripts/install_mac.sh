#!/bin/bash

# Проверка наличия Homebrew
if ! command -v brew &> /dev/null
then
    echo "Homebrew не найден. Устанавливаем Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew уже установлен."
fi

# Проверка наличия Xcode
if ! xcode-select -p &> /dev/null
then
    echo "Xcode не установлен. Пожалуйста, установите полную версию Xcode из App Store."
    exit 1
else
    echo "Xcode установлен."
fi

# Установка Qt5 через Homebrew
if ! brew list | grep -q "qt@5"
then
    echo "Устанавливаем Qt5..."
    brew install qt@5
else
    echo "Qt5 уже установлен."
fi

# Проверка и установка CMake через Homebrew
if ! command -v cmake &> /dev/null
then
    echo "Устанавливаем CMake..."
    brew install cmake
else
    echo "CMake уже установлен."
fi

# Проверка и установка OpenMP через Homebrew
if ! brew list | grep -q "libomp"
then
    echo "Устанавливаем OpenMP (libomp)..."
    brew install libomp
else
    echo "OpenMP (libomp) уже установлен."
fi

# # Проверка наличия Python и установка, если требуется (для работы с некоторыми инструментами CMake)
# if ! command -v python3 &> /dev/null
# then
#     echo "Устанавливаем Python3..."
#     brew install python
# else
#     echo "Python3 уже установлен."
# fi

# Проверка наличия ARM NEON библиотеки (ARM-оптимизации)
# Примечание: NEON является встроенной функцией процессоров ARM, поэтому библиотеку можно не устанавливать, 
# но мы проверим наличие необходимых флагов и совместимость с архитектурой.
if [[ $(uname -m) == "arm64" ]]
then
    echo "Архитектура ARM обнаружена. Будет использоваться поддержка NEON для SIMD."
else
    echo "Архитектура не ARM, NEON SIMD не доступен."
fi

echo "Установка завершена. Используйте Homebrew для управления установленными зависимостями."

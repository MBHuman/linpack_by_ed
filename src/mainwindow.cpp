#include "mainwindow.h"

#include <QDateTime>
#include <QFile>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QtConcurrent/QtConcurrent>

#include "linpack.h"
#ifdef __APPLE__
#include "gpu/metal_osx/linpack_gpu.h"  // Подключение GPU-версии Linpack (использует Metal или Cuda)
#else
#include "gpu/cuda/linpack_gpu.h"
#endif
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);

  // LAPACK: Создание и инициализация графика и виджета графика
  lapackChart = new QChart();
  lapackSeries = new QLineSeries();
  lapackChart->addSeries(lapackSeries);
  lapackChart->setTitle("Производительность Linpack");

  // Настройка осей для LAPACK
  lapackAxisX = new QValueAxis();
  lapackAxisY = new QValueAxis();
  lapackAxisX->setTitleText("Размер матрицы");
  lapackAxisY->setTitleText("GFLOPS");
  lapackChart->addAxis(lapackAxisX, Qt::AlignBottom);
  lapackChart->addAxis(lapackAxisY, Qt::AlignLeft);
  lapackSeries->attachAxis(lapackAxisX);
  lapackSeries->attachAxis(lapackAxisY);

  ui->lapackChartView->setChart(lapackChart);

  // Metal (GPU): Создание и инициализация графика и виджета графика
  gpuChart = new QChart();
  gpuSeries = new QLineSeries();
  gpuChart->addSeries(gpuSeries);
  gpuChart->setTitle("Производительность Metal (GPU)");

  // Настройка осей для Metal (GPU)
  gpuAxisX = new QValueAxis();
  gpuAxisY = new QValueAxis();
  gpuAxisX->setTitleText("Размер матрицы");
  gpuAxisY->setTitleText("GFLOPS");
  gpuChart->addAxis(gpuAxisX, Qt::AlignBottom);
  gpuChart->addAxis(gpuAxisY, Qt::AlignLeft);
  gpuSeries->attachAxis(gpuAxisX);
  gpuSeries->attachAxis(gpuAxisY);

  ui->gpuChartView->setChart(gpuChart);

  // Соединяем сигналы и слоты для кнопок LAPACK
  connect(ui->startLapackButton, &QPushButton::clicked, this,
          &MainWindow::runLinpackWithDifferentSizes);
  connect(ui->lapackClearButton, &QPushButton::clicked, this,
          &MainWindow::clearLapackChart);
  connect(ui->lapackSaveButton, &QPushButton::clicked, this,
          &MainWindow::saveResultsToFile);
  connect(ui->lapackLoadButton, &QPushButton::clicked, this,
          &MainWindow::loadResultsFromFile);
  connect(ui->lapackManualEntryButton, &QPushButton::clicked, this,
          &MainWindow::addManualEntry);

  // Соединяем сигналы и слоты для кнопок GPU
  connect(ui->startGpuButton, &QPushButton::clicked, this,
          &MainWindow::runLinpackGpuWithDifferentSizes);
  connect(ui->gpuClearButton, &QPushButton::clicked, this,
          &MainWindow::clearGpuChart);
  connect(ui->gpuSaveButton, &QPushButton::clicked, this,
          &MainWindow::saveResultsToFile);
  connect(ui->gpuLoadButton, &QPushButton::clicked, this,
          &MainWindow::loadResultsFromFile);
  connect(ui->gpuManualEntryButton, &QPushButton::clicked, this,
          &MainWindow::addManualEntry);

  // Соединение с watcher для обработки завершения вычислений
  connect(&futureWatcher, &QFutureWatcher<void>::finished, this,
          &MainWindow::handleResults);

  // Обработка переключения вкладок
  connect(ui->tabWidget, &QTabWidget::currentChanged, this,
          &MainWindow::onTabChanged);

  // Установка начального состояния интерфейса
  onTabChanged(0);  // Начальная вкладка - LAPACK
}

MainWindow::~MainWindow() {
  delete ui;
  delete lapackChart;
  delete lapackAxisX;
  delete lapackAxisY;
  delete gpuChart;
  delete gpuAxisX;
  delete gpuAxisY;
}

void MainWindow::runLinpackWithDifferentSizes() {
  bool ok;
  initialMatrixSize = QInputDialog::getInt(
      this, "Начальный размер матрицы",
      "Введите начальный размер матрицы:", 100, 1, 10000, 1, &ok);
  if (!ok) return;

  finalMatrixSize = QInputDialog::getInt(
      this, "Конечный размер матрицы", "Введите конечный размер матрицы:", 500,
      initialMatrixSize + 1, 20000, 1, &ok);
  if (!ok || finalMatrixSize <= initialMatrixSize) {
    QMessageBox::warning(
        this, "Ошибка",
        "Конечный размер матрицы должен быть больше начального.");
    return;
  }

  iterations =
      QInputDialog::getInt(this, "Количество итераций",
                           "Введите количество итераций:", 2, 2, 100, 1, &ok);
  if (!ok) return;

  lapackSeries->clear();
  ui->lapackProgressBar->setMinimum(0);
  ui->lapackProgressBar->setMaximum(100);
  ui->lapackProgressBar->setValue(0);

  step = (finalMatrixSize - initialMatrixSize) / (iterations - 1);

  auto future = QtConcurrent::run([=]() {
    for (int i = 0; i < iterations; ++i) {
      int size = initialMatrixSize + i * step;
      if (i == iterations - 1) size = finalMatrixSize;

      double elapsed_time = 0.0;
      double norma = 0.0;

      Linpack linpack(size, 1);
      linpack.runTest(elapsed_time, norma);

      double gflops = (2.0 / 3.0) * size * size * size / (elapsed_time * 1e9);

      QMetaObject::invokeMethod(
          this,
          [=]() {
            lapackSeries->append(size, gflops);

            ui->lapackPerformanceLabel->setText(
                QString("Оценочная производительность: %1 GFLOPS").arg(gflops));
            ui->lapackParrotsLabel->setText(
                QString("Количество попугаев: %1")
                    .arg(static_cast<int>(norma * 10)));

            lapackAxisX->setRange(
                initialMatrixSize,
                std::max(lapackAxisX->max(), static_cast<double>(size)));
            lapackAxisY->setRange(0, std::max(lapackAxisY->max(), gflops));

            int rowCount = ui->lapackHistoryTable->rowCount();
            ui->lapackHistoryTable->insertRow(rowCount);
            ui->lapackHistoryTable->setItem(
                rowCount, 0,
                new QTableWidgetItem(QDateTime::currentDateTime().toString()));
            ui->lapackHistoryTable->setItem(
                rowCount, 1, new QTableWidgetItem(QString::number(size)));
            ui->lapackHistoryTable->setItem(
                rowCount, 2, new QTableWidgetItem(QString::number(gflops)));

            int progressValue = static_cast<int>(
                (static_cast<double>(i + 1) / iterations) * 100);
            ui->lapackProgressBar->setValue(progressValue);
          },
          Qt::QueuedConnection);
    }
  });

  futureWatcher.setFuture(future);
}

void MainWindow::runLinpackGpuWithDifferentSizes() {
  // Аналогичная реализация для GPU
  // Используйте gpuSeries, gpuAxisX, gpuAxisY, ui->gpuProgressBar и другие
  // элементы интерфейса GPU

  bool ok;
  initialMatrixSize = QInputDialog::getInt(
      this, "Начальный размер матрицы",
      "Введите начальный размер матрицы:", 100, 1, 10000, 1, &ok);
  if (!ok) return;

  finalMatrixSize = QInputDialog::getInt(
      this, "Конечный размер матрицы", "Введите конечный размер матрицы:", 500,
      initialMatrixSize + 1, 20000, 1, &ok);
  if (!ok || finalMatrixSize <= initialMatrixSize) {
    QMessageBox::warning(
        this, "Ошибка",
        "Конечный размер матрицы должен быть больше начального.");
    return;
  }

  iterations =
      QInputDialog::getInt(this, "Количество итераций",
                           "Введите количество итераций:", 2, 2, 100, 1, &ok);
  if (!ok) return;

  gpuSeries->clear();
  ui->gpuProgressBar->setMinimum(0);
  ui->gpuProgressBar->setMaximum(100);
  ui->gpuProgressBar->setValue(0);

  step = (finalMatrixSize - initialMatrixSize) / (iterations - 1);

  auto future = QtConcurrent::run([=]() {
    for (int i = 0; i < iterations; ++i) {
      int size = initialMatrixSize + i * step;
      if (i == iterations - 1) size = finalMatrixSize;

      double elapsed_time = 0.0;
      double norma = 0.0;

      LinpackGPU linpack(size, 1);
      linpack.runTest(elapsed_time, norma);

      double gflops = (2.0 / 3.0) * size * size * size / (elapsed_time * 1e9);

      QMetaObject::invokeMethod(
          this,
          [=]() {
            gpuSeries->append(size, gflops);

            ui->gpuPerformanceLabel->setText(
                QString("Оценочная производительность: %1 GFLOPS").arg(gflops));
            ui->gpuParrotsLabel->setText(
                QString("Количество попугаев: %1")
                    .arg(static_cast<int>(norma * 10)));

            gpuAxisX->setRange(
                initialMatrixSize,
                std::max(gpuAxisX->max(), static_cast<double>(size)));
            gpuAxisY->setRange(0, std::max(gpuAxisY->max(), gflops));

            int rowCount = ui->gpuHistoryTable->rowCount();
            ui->gpuHistoryTable->insertRow(rowCount);
            ui->gpuHistoryTable->setItem(
                rowCount, 0,
                new QTableWidgetItem(QDateTime::currentDateTime().toString()));
            ui->gpuHistoryTable->setItem(
                rowCount, 1, new QTableWidgetItem(QString::number(size)));
            ui->gpuHistoryTable->setItem(
                rowCount, 2, new QTableWidgetItem(QString::number(gflops)));

            int progressValue = static_cast<int>(
                (static_cast<double>(i + 1) / iterations) * 100);
            ui->gpuProgressBar->setValue(progressValue);
          },
          Qt::QueuedConnection);
    }
  });

  futureWatcher.setFuture(future);
}

void MainWindow::clearLapackChart() {
  lapackSeries->clear();
  lapackAxisX->setRange(0, 1);
  lapackAxisY->setRange(0, 1);
}

void MainWindow::clearGpuChart() {
  gpuSeries->clear();
  gpuAxisX->setRange(0, 1);
  gpuAxisY->setRange(0, 1);
}

void MainWindow::handleResults() {
  QMessageBox::information(this, "Готово", "Тестирование завершено!");
}

void MainWindow::addManualEntry() {
  bool ok;
  int size =
      QInputDialog::getInt(this, "Добавить запись",
                           "Введите размер матрицы:", 500, 1, 10000, 1, &ok);
  if (!ok) return;

  double gflops = QInputDialog::getDouble(
      this, "Добавить запись", "Введите производительность (GFLOPS):", 0.0, 0.0,
      100.0, 2, &ok);
  if (!ok) return;

  if (ui->tabWidget->currentIndex() == 0) {
    lapackSeries->append(size, gflops);
    int rowCount = ui->lapackHistoryTable->rowCount();
    ui->lapackHistoryTable->insertRow(rowCount);
    ui->lapackHistoryTable->setItem(
        rowCount, 0,
        new QTableWidgetItem(QDateTime::currentDateTime().toString()));
    ui->lapackHistoryTable->setItem(
        rowCount, 1, new QTableWidgetItem(QString::number(size)));
    ui->lapackHistoryTable->setItem(
        rowCount, 2, new QTableWidgetItem(QString::number(gflops)));
    lapackAxisX->setRange(
        initialMatrixSize,
        std::max(lapackAxisX->max(), static_cast<double>(size)));
    lapackAxisY->setRange(0, std::max(lapackAxisY->max(), gflops));
  } else {
    gpuSeries->append(size, gflops);
    int rowCount = ui->gpuHistoryTable->rowCount();
    ui->gpuHistoryTable->insertRow(rowCount);
    ui->gpuHistoryTable->setItem(
        rowCount, 0,
        new QTableWidgetItem(QDateTime::currentDateTime().toString()));
    ui->gpuHistoryTable->setItem(rowCount, 1,
                                 new QTableWidgetItem(QString::number(size)));
    ui->gpuHistoryTable->setItem(rowCount, 2,
                                 new QTableWidgetItem(QString::number(gflops)));
    gpuAxisX->setRange(initialMatrixSize,
                       std::max(gpuAxisX->max(), static_cast<double>(size)));
    gpuAxisY->setRange(0, std::max(gpuAxisY->max(), gflops));
  }
}

void MainWindow::onTabChanged(int index) {
  if (index == 0) {
    // Активна вкладка LAPACK
    ui->startLapackButton->setVisible(true);
    ui->lapackClearButton->setVisible(true);
    ui->lapackSaveButton->setVisible(true);
    ui->lapackLoadButton->setVisible(true);
    ui->lapackManualEntryButton->setVisible(true);
    ui->lapackProgressBar->setVisible(true);
    ui->lapackPerformanceLabel->setVisible(true);
    ui->lapackParrotsLabel->setVisible(true);

    // Скрываем элементы GPU
    ui->startGpuButton->setVisible(false);
    ui->gpuClearButton->setVisible(false);
    ui->gpuSaveButton->setVisible(false);
    ui->gpuLoadButton->setVisible(false);
    ui->gpuManualEntryButton->setVisible(false);
    ui->gpuProgressBar->setVisible(false);
    ui->gpuPerformanceLabel->setVisible(false);
    ui->gpuParrotsLabel->setVisible(false);
  } else if (index == 1) {
    // Активна вкладка GPU
    ui->startLapackButton->setVisible(false);
    ui->lapackClearButton->setVisible(false);
    ui->lapackSaveButton->setVisible(false);
    ui->lapackLoadButton->setVisible(false);
    ui->lapackManualEntryButton->setVisible(false);
    ui->lapackProgressBar->setVisible(false);
    ui->lapackPerformanceLabel->setVisible(false);
    ui->lapackParrotsLabel->setVisible(false);

    // Показываем элементы GPU
    ui->startGpuButton->setVisible(true);
    ui->gpuClearButton->setVisible(true);
    ui->gpuSaveButton->setVisible(true);
    ui->gpuLoadButton->setVisible(true);
    ui->gpuManualEntryButton->setVisible(true);
    ui->gpuProgressBar->setVisible(true);
    ui->gpuPerformanceLabel->setVisible(true);
    ui->gpuParrotsLabel->setVisible(true);
  }
}

// Реализация метода для сохранения результатов в файл
void MainWindow::saveResultsToFile() {
  QString fileName = QFileDialog::getSaveFileName(this, "Сохранить результаты",
                                                  "", "CSV Files (*.csv)");
  if (!fileName.isEmpty()) {
    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly)) {
      QTextStream out(&file);
      out << "Дата,Размер Матрицы,Производительность (GFLOPS)\n";

      QTableWidget *historyTable = nullptr;
      if (ui->tabWidget->currentIndex() == 0) {
        historyTable = ui->lapackHistoryTable;
      } else if (ui->tabWidget->currentIndex() == 1) {
        historyTable = ui->gpuHistoryTable;
      }

      if (historyTable) {
        for (int i = 0; i < historyTable->rowCount(); ++i) {
          out << historyTable->item(i, 0)->text() << ","
              << historyTable->item(i, 1)->text() << ","
              << historyTable->item(i, 2)->text() << "\n";
        }
      }

      file.close();
    } else {
      QMessageBox::warning(this, "Ошибка",
                           "Не удалось открыть файл для записи.");
    }
  }
}

// Реализация метода для загрузки результатов из файла
void MainWindow::loadResultsFromFile() {
  QString fileName = QFileDialog::getOpenFileName(this, "Загрузить результаты",
                                                  "", "CSV Files (*.csv)");
  if (!fileName.isEmpty()) {
    QFile file(fileName);
    if (file.open(QIODevice::ReadOnly)) {
      QTextStream in(&file);
      QStringList header = in.readLine().split(",");  // Пропустить заголовок

      QTableWidget *historyTable = nullptr;
      QLineSeries *series = nullptr;
      QValueAxis *axisX = nullptr;
      QValueAxis *axisY = nullptr;

      if (ui->tabWidget->currentIndex() == 0) {
        historyTable = ui->lapackHistoryTable;
        series = lapackSeries;
        axisX = lapackAxisX;
        axisY = lapackAxisY;
      } else if (ui->tabWidget->currentIndex() == 1) {
        historyTable = ui->gpuHistoryTable;
        series = gpuSeries;
        axisX = gpuAxisX;
        axisY = gpuAxisY;
      }

      if (historyTable && series && axisX && axisY) {
        historyTable->setRowCount(0);
        series->clear();

        while (!in.atEnd()) {
          QString line = in.readLine();
          QStringList fields = line.split(",");
          if (fields.size() == 3) {
            int rowCount = historyTable->rowCount();
            historyTable->insertRow(rowCount);

            historyTable->setItem(rowCount, 0,
                                  new QTableWidgetItem(fields.at(0)));
            historyTable->setItem(rowCount, 1,
                                  new QTableWidgetItem(fields.at(1)));
            historyTable->setItem(rowCount, 2,
                                  new QTableWidgetItem(fields.at(2)));

            bool ok;
            int size = fields.at(1).toInt(&ok);
            double gflops = fields.at(2).toDouble(&ok);
            if (ok) {
              series->append(size, gflops);
            }
          }
        }

        // Обновление диапазонов осей
        if (series->count() > 0) {
          axisX->setRange(0, historyTable->rowCount() > 0
                                 ? series->at(series->count() - 1).x()
                                 : 1);
          axisY->setRange(0, historyTable->rowCount() > 0
                                 ? series->at(series->count() - 1).y()
                                 : 1);
        }
      }

      file.close();
    } else {
      QMessageBox::warning(this, "Ошибка",
                           "Не удалось открыть файл для чтения.");
    }
  }
}
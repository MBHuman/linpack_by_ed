#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QFutureWatcher>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

using namespace QtCharts;

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private slots:
  void runLinpackWithDifferentSizes();
  void runLinpackGpuWithDifferentSizes();
  void clearLapackChart();
  void clearGpuChart();
  void onTabChanged(int index);
  void handleResults();
  void addManualEntry();
  void saveResultsToFile();  // Добавляем метод для сохранения результатов в файл
  void loadResultsFromFile(); // Добавляем метод для загрузки результатов из файла


private:
  Ui::MainWindow *ui;

  // Локальные переменные для графиков и осей LAPACK
  QChart *lapackChart;
  QLineSeries *lapackSeries;
  QValueAxis *lapackAxisX;
  QValueAxis *lapackAxisY;

  // Локальные переменные для графиков и осей GPU
  QChart *gpuChart;
  QLineSeries *gpuSeries;
  QValueAxis *gpuAxisX;
  QValueAxis *gpuAxisY;

  QFutureWatcher<void> futureWatcher;

  // Переменные для выполнения тестов
  int initialMatrixSize;
  int finalMatrixSize;
  int iterations;
  int step; // Добавляем переменную для шага
};

#endif // MAINWINDOW_H

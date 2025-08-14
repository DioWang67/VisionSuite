import sys
import logging
import argparse
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QCheckBox,
    QTextEdit,
    QMessageBox,
    QHBoxLayout,
)

from main_pipeline import load_config, setup_logging, run_pipeline


class QTextEditLogger(logging.Handler):
    """將日誌輸出導向 QTextEdit"""

    def __init__(self, widget: QTextEdit):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.append(msg)


class VisionSuiteUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VisionSuite 圖形介面")
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # 配置文件選擇
        config_layout = QHBoxLayout()
        self.config_label = QLabel("使用預設 config.yaml")
        config_btn = QPushButton("選擇配置檔")
        config_btn.clicked.connect(self.select_config)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(config_btn)
        layout.addLayout(config_layout)

        # 任務勾選框與對應路徑
        self.task_paths = {
            "format_conversion": Path(__file__).with_name("image_format_converter.py"),
            "anomaly_detection": Path(__file__).with_name("anomaly_mask_generator.py"),
            "yolo_augmentation": Path(__file__).with_name("yolo_data_augmentor.py"),
            "image_augmentation": Path(__file__).with_name("image_augmentor.py"),
            "dataset_splitter": Path(__file__).with_name("dataset_splitter.py"),
        }
        self.tasks = {
            name: QCheckBox(label)
            for name, label in {
                "format_conversion": "格式轉換",
                "anomaly_detection": "異常檢測",
                "yolo_augmentation": "YOLO 增強",
                "image_augmentation": "圖像增強",
                "dataset_splitter": "資料集分割",
            }.items()
        }
        for name, cb in self.tasks.items():
            cb.setToolTip(str(self.task_paths[name]))
            layout.addWidget(cb)

        # 顯示功能路徑按鈕
        path_btn = QPushButton("顯示功能路徑")
        path_btn.clicked.connect(self.show_task_paths)
        layout.addWidget(path_btn)

        # 執行按鈕
        run_btn = QPushButton("執行流程")
        run_btn.clicked.connect(self.execute_pipeline)
        layout.addWidget(run_btn)

        # 日誌輸出區域
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.config_path = "config.yaml"

    def select_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置檔", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            self.config_path = path
            self.config_label.setText(path)

    def execute_pipeline(self) -> None:
        selected_tasks = [name for name, cb in self.tasks.items() if cb.isChecked()]
        if not selected_tasks:
            QMessageBox.warning(self, "提示", "請至少選擇一個任務")
            return
        try:
            config = load_config(self.config_path)
            logger = setup_logging(config["pipeline"]["log_file"])
            text_handler = QTextEditLogger(self.log_output)
            text_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(text_handler)

            args = argparse.Namespace(
                config=self.config_path, input_format=None, output_format=None
            )

            run_pipeline(selected_tasks, config, logger, args)
            QMessageBox.information(self, "完成", "流程執行完成")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    def show_task_paths(self) -> None:
        paths_info = "\n".join(
            f"{cb.text()}: {self.task_paths[name]}" for name, cb in self.tasks.items()
        )
        QMessageBox.information(self, "功能路徑", paths_info)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = VisionSuiteUI()
    ui.show()
    sys.exit(app.exec_())

#!/bin/bash

# Script để chạy test_vision_pipeline.py với conda environment
# Cách sử dụng:
#   ./test_vision_pipeline.bash [tên_conda_env] [đường_dẫn_file]

# Tên conda environment (thay đổi theo môi trường của bạn)
CONDA_ENV="${1:-ducmb}"  # Mặc định là "ragflow" nếu không truyền tham số

# Đường dẫn file input (thay đổi theo file của bạn)
INPUT_FILE="${2:-/home/ducmb/Projects/ragflow2/deepdoc/vision/test.pdf}"

# Đường dẫn đến root của project (ragflow)
PROJECT_ROOT="/home/ducmb/Projects/ragflow2"

# Đường dẫn đến script Python
SCRIPT_PATH="$PROJECT_ROOT/deepdoc/test_vision_pipeline.py"

# Kiểm tra xem conda có được cài đặt không
if ! command -v conda &> /dev/null; then
    echo "Lỗi: conda không được tìm thấy. Vui lòng cài đặt conda hoặc thêm vào PATH."
    exit 1
fi

# Initialize conda (nếu chưa được init trong shell này)
eval "$(conda shell.bash hook)"

# Activate conda environment
echo "Đang kích hoạt conda environment: $CONDA_ENV"
if ! conda activate "$CONDA_ENV"; then
    echo "Lỗi: Không thể kích hoạt conda environment '$CONDA_ENV'"
    echo "Vui lòng kiểm tra tên environment hoặc tạo mới bằng: conda create -n $CONDA_ENV python=3.10"
    exit 1
fi

# Kiểm tra Python có sẵn không
if ! command -v python &> /dev/null; then
    echo "Lỗi: python không được tìm thấy trong environment $CONDA_ENV"
    exit 1
fi

# Kiểm tra file script có tồn tại không
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Lỗi: Không tìm thấy file script tại $SCRIPT_PATH"
    exit 1
fi

# Set PYTHONPATH để Python có thể tìm thấy các module (api, rag, deepdoc, etc.)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Chuyển đến thư mục deepdoc để import 'from vision' hoạt động đúng
cd "$PROJECT_ROOT/deepdoc"

# Chạy script Python (sử dụng đường dẫn tương đối từ deepdoc/)
echo "Đang chạy test_vision_pipeline.py với input: $INPUT_FILE"
echo "PYTHONPATH: $PYTHONPATH"
echo "Working directory: $(pwd)"
python test_vision_pipeline.py --inputs "$INPUT_FILE"

# Lưu exit code
EXIT_CODE=$?

exit $EXIT_CODE
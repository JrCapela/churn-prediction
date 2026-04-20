.PHONY: install install-dev train test test-unit test-integration run docker-build docker-run clean

# Instalar dependências de produção
install:
	pip install -r requirements.txt

# Instalar dependências de desenvolvimento
install-dev:
	pip install -r requirements-dev.txt

# Treinar o modelo
train:
	python -c "import sys; sys.path.append('.'); from src.train import train_all; train_all('data/raw/telco_churn.csv')"

# Rodar todos os testes
test:
	pytest tests/ -v --tb=short

# Rodar apenas testes unitários
test-unit:
	pytest tests/ -v -m "unit" --tb=short

# Rodar apenas testes de integração
test-integration:
	pytest tests/ -v -m "integration" --tb=short

# Subir a API localmente
run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Subir MLflow
mlflow:
	mlflow server --host 127.0.0.1 --port 5000

# Build da imagem Docker
docker-build:
	docker compose build

# Subir containers
docker-run:
	docker compose up

# Parar containers
docker-stop:
	docker compose down

# Push dos dados para o GCS
dvc-push:
	dvc push

# Pull dos dados do GCS
dvc-pull:
	dvc pull

# Limpar arquivos temporários
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
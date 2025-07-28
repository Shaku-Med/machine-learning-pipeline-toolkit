# ML Pipeline System

A complete Machine Learning pipeline system that takes you from raw data to deployed models. We built this to solve the common challenges of ML engineering - making it easier to build, test, deploy, and monitor machine learning systems in production.

## What This Is

Think of this as your all-in-one toolkit for machine learning. Whether you're just getting started with ML or you're managing production systems, this pipeline gives you everything you need. It handles the messy parts of ML engineering so you can focus on what matters - building great models.

## What You Get

**Data Handling**
- Clean up messy data automatically
- Build features without the headache
- Keep track of data quality
- Know when your data changes

**Model Building**
- Try different algorithms easily
- Find the best settings automatically
- Compare models side by side
- Know which model actually works

**Getting Models to Work**
- Turn your model into a web service
- Make predictions through a simple API
- Keep track of different model versions
- Scale up when you need to

**Keeping Things Running**
- Spot when your data starts changing
- Watch how your model performs
- Get alerts when things go wrong
- See what's happening under the hood

**Making Life Easier**
- Control everything from the command line
- Configure things without coding
- Generate reports automatically
- Test everything thoroughly

## Project Structure

```
ml_pipeline/
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   └── env.example        # Environment variables template
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── models/                 # Trained models
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model training and evaluation
│   ├── deployment/        # Model serving
│   ├── monitoring/        # Performance monitoring
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── logs/                  # Application logs
├── plots/                 # Generated visualizations
├── requirements.txt       # Python dependencies
├── requirements-minimal.txt # Minimal dependencies
├── cli.py                 # Command-line interface
├── demo_simple.py         # Demo script
├── test_pipeline.py       # Core functionality test
├── quick_start.py         # Quick start guide
└── run_all_tests.py       # Comprehensive test suite
```

## Getting Started

### What You Need

- Python 3.8 or newer (most people have this)
- pip (comes with Python)

### Setting Things Up

1. **Get the code**:
   ```bash
   git clone <repository-url>
   cd ml_pipeline
   ```

2. **Install what you need**:
   ```bash
   pip install -r requirements.txt
   ```

   If you want to start small (just the basics):
   ```bash
   pip install -r requirements-minimal.txt
   ```

3. **Set up your environment** (optional, but recommended):
   ```bash
   cp env.example .env
   # Open .env and change any settings you want
   ```

### The Easy Way

Want to see everything working right now? Run this:

```bash
python quick_start.py
```

This will:
- Check that everything is set up correctly
- Create some sample data for you
- Run the whole pipeline
- Show you what it created

### The Step-by-Step Way

If you like to understand what's happening at each step:

1. **Make sure everything works**:
   ```bash
   python test_imports.py
   ```

2. **Create data and run the pipeline**:
   ```bash
   python cli.py run-pipeline --create-sample
   ```

3. **Just train some models**:
   ```bash
   python cli.py train-models
   ```

4. **Create some charts**:
   ```bash
   python cli.py visualize
   ```

## Using the Command Line

Everything can be controlled from the command line. Here are the main things you'll use:

```bash
# See what commands are available
python cli.py --help

# Run the whole pipeline (creates data, trains models, etc.)
python cli.py run-pipeline --create-sample

# Just work with data
python cli.py data-pipeline -d data/sample_data.csv -t target

# Just train models
python cli.py train-models

# Start the model server
python cli.py serve-model

# Check what's working
python cli.py status

# Run tests
python cli.py test

# Create charts and reports
python cli.py visualize

# Clean up your code
python cli.py format-code
```

## Configuration

All the settings are in YAML files in the `config/` folder. The main one is `config.yaml` - that's where you'll change most things.

You can also use environment variables to override any setting.

Here's what you can configure:
- **App**: Basic settings like ports and timeouts
- **Data**: How to process and clean your data
- **Model**: Which algorithms to try
- **Training**: How to find the best model settings
- **Evaluation**: What metrics matter to you
- **Deployment**: How to serve your models
- **Monitoring**: When to alert you about problems
- **Logging**: How much detail to record

## API and Web Interface

Once you start the model server, you can access:

- **API docs**: http://localhost:8000/docs (see all available endpoints)
- **Health check**: http://localhost:8000/health (make sure it's working)
- **Make predictions**: http://localhost:8000/predict (send data, get predictions)

## Testing Everything

We've included tests to make sure everything works:

```bash
# Run all the tests
python run_all_tests.py

# Test the core functionality
python test_pipeline.py

# Test that all imports work
python test_imports.py

# Test the command line interface
python test_cli.py
```

## For Developers

### Keeping Code Clean

```bash
# Format your code automatically
python cli.py format-code

# Run the test suite
python cli.py test

# Check types (if you have mypy installed)
mypy src/
```

### How the Code is Organized

- **src/**: All the main code
  - **data/**: Everything about processing data
  - **models/**: Training and evaluating models
  - **deployment/**: Making models available via API
  - **monitoring/**: Watching for problems
  - **utils/**: Helper functions and configuration

- **tests/**: Tests to make sure everything works
- **config/**: Settings and configuration files
- **data/**: Where your data lives (raw and processed)
- **models/**: Where trained models are saved
- **logs/**: Records of what happened
- **plots/**: Charts and visualizations

## When Things Go Wrong

### Common Problems

1. **Import errors**: You probably need to install some packages - try `pip install -r requirements.txt`
2. **Model not found**: You need to train models first - run `python cli.py train-models`
3. **Data not found**: Create some sample data first - run `python cli.py run-pipeline --create-sample`
4. **Port already in use**: Something else is using port 8000 - change it in the config file

### Getting Help

1. Look in the `logs/` folder - there might be error messages there
2. Run `python cli.py status` to see what's working and what isn't
3. Try `python test_imports.py` to check if everything is set up correctly
4. Check the settings in `config/config.yaml` - maybe something needs to be changed

## Contributing

Want to help make this better? Here's how:

1. Fork the repository (make your own copy)
2. Create a branch for your changes
3. Make your improvements
4. Add tests for any new features
5. Make sure all tests still pass
6. Send us a pull request

## License

This is open source under the MIT License - you can use it, modify it, and share it.

## Getting Help

If you run into problems or have questions:

1. Check if someone else already asked the same question
2. Look through this documentation
3. Create a new issue and tell us what's happening
4. Include any error messages you're seeing 
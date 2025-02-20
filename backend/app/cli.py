import click
from app.main import app

@click.group()
def cli():
    pass

@cli.command()
def run():
    """Run the FastAPI application."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    cli()

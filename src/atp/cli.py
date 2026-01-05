from __future__ import annotations
import asyncio
from pathlib import Path
import typer
from rich import print

from atp.rag.rag_core import build_vectorstore, answer_query
from atp.web.scrape import scrape_url

app = typer.Typer(no_args_is_help=True)

DEFAULT_CHROMA_DIR = Path("data/chroma")


@app.command()
def rag_ingest(
    docs_dir: Path = typer.Option(Path("docs"), help="Thư mục chứa PDF"),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Chroma persist dir"),
    embed_model: str = typer.Option("embeddinggemma", help="Ollama embedding model"),
):
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise typer.BadParameter(f"Không thấy PDF trong {docs_dir}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    n = build_vectorstore(pdfs, chroma_dir, embed_model=embed_model)
    print(f"[green]OK[/green] Indexed {n} chunks into {chroma_dir}")


@app.command()
def rag_query(
    question: str = typer.Argument(..., help="Câu hỏi"),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Chroma persist dir"),
    embed_model: str = typer.Option("embeddinggemma", help="Ollama embedding model"),
    chat_model: str = typer.Option("llama3.1", help="Ollama chat model"),
    top_k: int = typer.Option(4, help="Số chunk truy hồi"),
):
    ans = answer_query(
        question=question,
        persist_dir=chroma_dir,
        embed_model=embed_model,
        chat_model=chat_model,
        top_k=top_k,
    )
    print(ans)


@app.command()
def web_scrape(
    url: str = typer.Argument(..., help="URL cần lấy"),
    out_dir: Path = typer.Option(Path("outputs"), help="Thư mục output"),
    allowed_domain: list[str] = typer.Option(None, help="Domain allowlist (lặp nhiều lần)"),
    headless: bool = typer.Option(True, help="Chạy headless"),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    async def _run():
        r = await scrape_url(url, allowed_domains=allowed_domain, headless=headless)
        (out_dir / "page.html").write_text(r.html, encoding="utf-8")
        (out_dir / "page.txt").write_text(r.text, encoding="utf-8")
        print(f"[green]OK[/green] Saved HTML/TEXT to {out_dir}")

    asyncio.run(_run())

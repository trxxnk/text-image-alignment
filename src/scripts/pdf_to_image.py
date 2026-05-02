import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pymupdf


def process_page(args):
    pdf_path, pg_num, idx, save_dir, name_template = args

    try:
        doc = pymupdf.open(pdf_path)
        page = doc[pg_num]

        dpi = int(100 + random.expovariate(1/20))
        dpi = min(dpi, 200)

        pix = page.get_pixmap(dpi=dpi)

        name = name_template.format(idx)
        save_path = save_dir / name

        pix.save(str(save_path))

        doc.close()

        return 1

    except Exception as e:
        print(f"Error: {pdf_path}, page {pg_num}: {e}")
        return 0


def generate_images_parallel(pdf_files, save_dir, name_template):
    tasks = []
    idx = 0

    for pdf_file in pdf_files:
        with pymupdf.open(pdf_file) as doc:
            n_pages = doc.page_count

        start = 10
        end = max(n_pages - 10, 10)

        for pg_num in range(start, end):
            tasks.append((pdf_file, pg_num, idx, save_dir, name_template))
            idx += 1

    # Параллельная обработка
    with ProcessPoolExecutor(max_workers=12) as executor:
        list(tqdm(
            executor.map(process_page, tasks),
            total=len(tasks),
            desc="Генерация изображений"
        ))


if __name__ == "__main__":
    import os, glob

    pdf_files = glob.glob('data/pdf/*.pdf')

    name_temp = "generated_{:05}.png"
    save_dir = Path("./data/raw/v3")
    os.makedirs(save_dir, exist_ok=True)

    generate_images_parallel(pdf_files, save_dir, name_temp)

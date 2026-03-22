"""
Конвертер файлов Neurosoft .EEG -> CSV

Формат файла D0000001.EEG (Neurosoft Neuron-Spectrum):
- Заголовок: метаданные пациента, параметры записи, описание каналов
- Данные: 24-битные знаковые целые, little-endian, каналы чередуются
- Частота дискретизации: 250 Гц

Использование:
    python main.py                          # конвертировать весь файл
    python main.py --seconds 60             # только первые 60 секунд
    python main.py --eeg-only               # только ЭЭГ каналы (без Bio, A1, A2, синхро)
    python main.py --raw                    # сырые значения АЦП (без калибровки)
    python main.py --output my_data.csv     # указать имя выходного файла
"""

import struct
import os
import sys
import time
import argparse


# ─── Константы формата Neurosoft ────────────────────────────────────────────

HEADER_SRATE_OFFSET = 0x04         # uint16: частота дискретизации
HEADER_NCHAN_OFFSET = 0x02         # uint16: количество каналов
HEADER_CHAN_INFO_OFFSET = 0x18     # uint32: смещение описания каналов
HEADER_DATA_OFFSET_FIELD = 0x1C   # uint32: смещение начала данных

CHAN_RECORD_SIZE = 0x40            # 64 байта на описание канала
CHAN_NAME_SIZE = 8                 # длина имени канала
CHAN_CAL_OFFSET = 0x18             # float32: калибровочный коэффициент (µV/ед)

BYTES_PER_SAMPLE = 3              # 24-битное разрешение АЦП


# ─── Функции чтения ─────────────────────────────────────────────────────────

def read_int24_le(data: bytes, offset: int) -> int:
    """Прочитать 24-битное знаковое целое, little-endian."""
    b0 = data[offset]
    b1 = data[offset + 1]
    b2 = data[offset + 2]
    val = b0 | (b1 << 8) | (b2 << 16)
    if val >= 0x800000:
        val -= 0x1000000
    return val


def parse_header(filepath: str) -> dict:
    """Разобрать заголовок файла .EEG и вернуть метаданные."""
    with open(filepath, 'rb') as f:
        header = f.read(0x1700)

    info = {}

    # Основные параметры
    info['n_channels'] = struct.unpack_from('<H', header, HEADER_NCHAN_OFFSET)[0]
    info['srate'] = struct.unpack_from('<H', header, HEADER_SRATE_OFFSET)[0]
    info['chan_info_offset'] = struct.unpack_from('<I', header, HEADER_CHAN_INFO_OFFSET)[0]
    info['data_offset'] = struct.unpack_from('<I', header, HEADER_DATA_OFFSET_FIELD)[0]

    # Метаданные исследования
    try:
        raw = header[0x90:0xC0].decode('cp1251')
        # Обрезать по первому нулю
        if '\x00' in raw:
            raw = raw[:raw.index('\x00')]
        info['study_name'] = raw.strip()
    except Exception:
        info['study_name'] = ''

    try:
        info['date'] = header[0xC4:0xCE].decode('ascii').rstrip('\x00').strip()
    except Exception:
        info['date'] = ''

    try:
        info['time'] = header[0xD0:0xD8].decode('ascii').rstrip('\x00').strip()
    except Exception:
        info['time'] = ''

    try:
        info['device'] = header[0xDA:0xEA].decode('ascii').rstrip('\x00').strip()
    except Exception:
        info['device'] = ''

    try:
        info['patient_birth'] = header[0x10E:0x118].decode('ascii').rstrip('\x00').strip()
    except Exception:
        info['patient_birth'] = ''

    try:
        info['patient_sex'] = header[0x118:0x11A].decode('ascii').rstrip('\x00').strip()
    except Exception:
        info['patient_sex'] = ''

    # Прочитать описания каналов
    channels = []
    ch_start = info['chan_info_offset']
    for i in range(info['n_channels']):
        off = ch_start + i * CHAN_RECORD_SIZE
        if off + CHAN_RECORD_SIZE > len(header):
            break

        # Имя канала - до первого нуля
        name = ''
        for j in range(CHAN_NAME_SIZE):
            b = header[off + j]
            if b == 0:
                break
            name += chr(b)

        cal = struct.unpack_from('<f', header, off + CHAN_CAL_OFFSET)[0]
        channels.append({'name': name, 'calibration': cal, 'index': i})

    info['channels'] = channels

    # Определить точное начало данных
    # Данные начинаются после всех описаний каналов и таблиц монтажа
    # Проверяем что data_offset валиден по паттерну LABEL=0
    data_offset = info['data_offset']
    record_size = info['n_channels'] * BYTES_PER_SAMPLE
    label_ch_idx = None
    for ch in channels:
        if ch['name'] == 'LABEL':
            label_ch_idx = ch['index']
            break

    if label_ch_idx is not None:
        # Проверяем что LABEL = 0 на первых сэмплах
        with open(filepath, 'rb') as f:
            f.seek(data_offset)
            test_data = f.read(record_size * 5)
            all_zero = True
            for s in range(5):
                base = s * record_size + label_ch_idx * BYTES_PER_SAMPLE
                if base + 3 <= len(test_data):
                    v = read_int24_le(test_data, base)
                    if v != 0:
                        all_zero = False
                        break
            if not all_zero:
                print(f"  ВНИМАНИЕ: LABEL канал не равен 0 при data_offset=0x{data_offset:X}.")
                print(f"  Пытаемся найти правильное смещение...")
                # Ищем правильное смещение по тройному нулю в позиции LABEL
                with open(filepath, 'rb') as f2:
                    f2.seek(data_offset)
                    search_data = f2.read(record_size * 3)
                    for trial_off in range(0, record_size):
                        ok = True
                        for s in range(3):
                            base = trial_off + s * record_size + label_ch_idx * BYTES_PER_SAMPLE
                            if base + 3 <= len(search_data):
                                v = read_int24_le(search_data, base)
                                if v != 0:
                                    ok = False
                                    break
                        if ok:
                            data_offset += trial_off
                            print(f"  Найдено смещение: 0x{data_offset:X}")
                            break

    info['data_offset'] = data_offset

    # Количество сэмплов
    file_size = os.path.getsize(filepath)
    data_bytes = file_size - data_offset
    info['n_samples'] = data_bytes // record_size
    info['record_size'] = record_size
    info['duration_sec'] = info['n_samples'] / info['srate']
    info['file_size'] = file_size

    return info


def print_info(info: dict):
    """Вывести информацию о файле."""
    print("=" * 60)
    print("  Информация о файле EEG (Neurosoft)")
    print("=" * 60)
    print(f"  Исследование  : {info['study_name']}")
    print(f"  Дата/время    : {info['date']} {info['time']}")
    print(f"  Аппарат       : {info['device']}")
    print(f"  Дата рождения : {info['patient_birth']}")
    print(f"  Пол           : {info['patient_sex']}")
    print(f"  Каналов       : {info['n_channels']}")
    print(f"  Частота (Гц)  : {info['srate']}")
    print(f"  Сэмплов       : {info['n_samples']:,}")
    print(f"  Длительность  : {info['duration_sec']:.1f} с ({info['duration_sec']/60:.1f} мин)")
    print(f"  Размер файла  : {info['file_size']:,} байт")
    print(f"  Смещение данных: 0x{info['data_offset']:X}")
    print()
    print("  Каналы:")
    for ch in info['channels']:
        print(f"    {ch['index']:2d}. {ch['name']:6s}  (калибровка: {ch['calibration']:.6f})")
    print("=" * 60)


def convert_to_csv(filepath: str, output: str, info: dict,
                   eeg_only: bool = False, raw: bool = False,
                   max_seconds: float = None):
    """Конвертировать .EEG файл в CSV."""

    n_channels = info['n_channels']
    srate = info['srate']
    channels = info['channels']
    data_offset = info['data_offset']
    record_size = info['record_size']
    n_samples = info['n_samples']

    if max_seconds is not None:
        n_samples = min(n_samples, int(max_seconds * srate))

    # Выбрать каналы для экспорта
    if eeg_only:
        # Исключить Bio, A1, A2, VSyn, ASyn, LABEL
        excluded = {'Bio1', 'Bio2', 'A1', 'A2', 'VSyn', 'ASyn', 'LABEL'}
        export_channels = [ch for ch in channels if ch['name'] not in excluded]
    else:
        export_channels = channels

    ch_indices = [ch['index'] for ch in export_channels]
    ch_names = [ch['name'] for ch in export_channels]
    ch_cals = [ch['calibration'] for ch in export_channels]

    print(f"\nЭкспорт в CSV:")
    print(f"  Каналов      : {len(export_channels)} из {n_channels}")
    print(f"  Сэмплов      : {n_samples:,}")
    print(f"  Формат данных: {'сырые (АЦП)' if raw else 'микровольты (µV)'}")
    print(f"  Файл         : {output}")

    # Буферизированная запись для производительности
    CHUNK_SAMPLES = 10000  # читать по 10000 сэмплов за раз
    chunk_bytes = CHUNK_SAMPLES * record_size

    start_time = time.time()
    samples_written = 0

    with open(filepath, 'rb') as fin, open(output, 'w') as fout:
        # Заголовок CSV
        header_line = 'time_s,' + ','.join(ch_names)
        fout.write(header_line + '\n')

        fin.seek(data_offset)

        while samples_written < n_samples:
            remaining = n_samples - samples_written
            to_read = min(CHUNK_SAMPLES, remaining)
            data = fin.read(to_read * record_size)

            if len(data) < to_read * record_size:
                to_read = len(data) // record_size
                if to_read == 0:
                    break

            lines = []
            for s in range(to_read):
                t = (samples_written + s) / srate
                values = []
                for ci, ch_idx in enumerate(ch_indices):
                    base = s * record_size + ch_idx * BYTES_PER_SAMPLE
                    raw_val = read_int24_le(data, base)
                    if raw:
                        values.append(str(raw_val))
                    else:
                        uv = raw_val * ch_cals[ci]
                        values.append(f'{uv:.2f}')

                line = f'{t:.4f},' + ','.join(values)
                lines.append(line)

            fout.write('\n'.join(lines) + '\n')
            samples_written += to_read

            # Прогресс
            elapsed = time.time() - start_time
            pct = samples_written / n_samples * 100
            if elapsed > 0:
                speed = samples_written / elapsed
                eta = (n_samples - samples_written) / speed if speed > 0 else 0
                print(f'\r  Прогресс: {pct:.1f}% ({samples_written:,}/{n_samples:,})'
                      f'  Скорость: {speed:,.0f} сэмплов/с  ETA: {eta:.0f}с', end='')

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output) / (1024 * 1024)

    print(f'\n\n  Готово!')
    print(f'  Записано      : {samples_written:,} сэмплов')
    print(f'  Время         : {elapsed:.1f} с')
    print(f'  Размер CSV    : {file_size_mb:.1f} МБ')


def main():
    parser = argparse.ArgumentParser(
        description='Конвертер Neurosoft .EEG файлов в CSV')
    parser.add_argument('input', nargs='?', default='D0000001.EEG',
                        help='Путь к .EEG файлу (по умолчанию: D0000001.EEG)')
    parser.add_argument('--output', '-o', default=None,
                        help='Имя выходного CSV файла')
    parser.add_argument('--eeg-only', action='store_true',
                        help='Только ЭЭГ каналы (без Bio, A1, A2, Sync, Label)')
    parser.add_argument('--raw', action='store_true',
                        help='Сырые значения АЦП без калибровки')
    parser.add_argument('--seconds', type=float, default=None,
                        help='Экспортировать только первые N секунд')
    parser.add_argument('--info', action='store_true',
                        help='Только показать информацию о файле')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Ошибка: файл не найден: {args.input}')
        sys.exit(1)

    # Разобрать заголовок
    info = parse_header(args.input)
    print_info(info)

    if args.info:
        return

    # Имя выходного файла
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        suffix = ''
        if args.eeg_only:
            suffix += '_eeg'
        if args.raw:
            suffix += '_raw'
        if args.seconds:
            suffix += f'_{int(args.seconds)}s'
        args.output = f'{base}{suffix}.csv'

    convert_to_csv(
        filepath=args.input,
        output=args.output,
        info=info,
        eeg_only=args.eeg_only,
        raw=args.raw,
        max_seconds=args.seconds,
    )


if __name__ == '__main__':
    main()

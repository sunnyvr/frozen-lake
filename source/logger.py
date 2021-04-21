OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL = range(8)


class Logger:

    def __init__(self, log_level=INFO):
        self.log_level = log_level
        self.index = 0
        self.index_show_modulo = 1

    @staticmethod
    def with_max_index(max_index, index_show_percent, level=INFO):
        level_map = {'off': OFF, 'fatal': FATAL, 'error': ERROR, 'warn': WARN,
                     'info': INFO, 'debug': DEBUG, 'trace': TRACE, 'all': ALL}

        if type(level) is str:
            level = level_map[level] if level in level_map else INFO

        logger = Logger(level)
        logger.configure_max_index(max_index, index_show_percent)
        return logger

    def configure_max_index(self, max_index, index_show_percent):
        calculated_modulo = int(max_index * index_show_percent)
        self.index_show_modulo = max(calculated_modulo, 1)

    def set_index(self, index):
        self.index = index

    def should_show_index(self):
        return self.index % self.index_show_modulo == 0

    def print_level(self, level, *args):
        if level <= self.log_level and self.should_show_index():
            print(*args)

    def fatal(self, *args):
        self.print_level(FATAL, *args)

    def error(self, *args):
        self.print_level(ERROR, *args)

    def warn(self, *args):
        self.print_level(WARN, *args)

    def info(self, *args):
        self.print_level(INFO, *args)

    def debug(self, *args):
        self.print_level(DEBUG, *args)

    def trace(self, *args):
        self.print_level(TRACE, *args)


def main():
    print(TRACE)


if __name__ == "__main__":
    main()

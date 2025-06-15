import os, sys

from pybind11_stubgen import *

# pyd 路径
cApiPath = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, cApiPath)

logger = logging.getLogger(__name__)

error_handlers_top = [
    LoggerData
]


class Parser(
    *error_handlers_top,  # type: ignore[misc]
    FixMissing__future__AnnotationsImport,
    FixMissing__all__Attribute,
    FixMissingNoneHashFieldAnnotation,
    FixMissingImports,
    FilterTypingModuleAttributes,
    FixPEP585CollectionNames,
    FixTypingTypeNames,
    FixScipyTypeArguments,
    FixMissingFixedSizeImport,
    FixMissingEnumMembersAnnotation,
    OverridePrintSafeValues,
    # *numpy_fixes,  # type: ignore[misc]
    FixNumpyDtype,
    FixNumpyArrayFlags,
    FixCurrentModulePrefixInTypeNames,
    FixBuiltinTypes,
    RewritePybind11EnumValueRepr,
    FilterClassMembers,
    ReplaceReadWritePropertyWithField,
    FilterInvalidIdentifiers,
    FixValueReprRandomAddress,
    FixRedundantBuiltinsAnnotation,
    FilterPybindInternals,
    FixRedundantMethodsFromBuiltinObject,
    RemoveSelfAnnotation,
    FixPybind11EnumStrDoc,
    ExtractSignaturesFromPybind11Docstrings,
    ParserDispatchMixin,
    BaseParser,
    # *error_handlers_bottom,  # type: ignore[misc]
):
    ...


def generate_pyi(module_name):
    try:
        exec("import %s" % module_name)
        w = Writer()
        parser = Parser()
        module = parser.handle_module(
            QualifiedName.from_str(module_name), importlib.import_module(module_name)
        )
        w.write_module(module, Printer(True), Path(cApiPath))
    except:
        import traceback
        traceback.print_exc()
        logger.error("导入模块[%s]失败" % module_name)


def generate_all():
    for file_name in os.listdir(cApiPath):

        if file_name.endswith("pyd"):
            module_name = file_name.split(".")[0]
            generate_pyi(module_name)


if __name__ == '__main__':
    pass
    generate_all()

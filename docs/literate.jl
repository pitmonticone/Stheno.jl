# Copied almost exactly from KernelFunctions.jl.

# Retrieve name of example and output directory
if length(ARGS) != 3
    error("please specify the name of the example and the output directory")
end
const EXAMPLEDIR = ARGS[1]
const EXAMPLESAVENAME = ARGS[2]
const OUTDIR = ARGS[3]

# Activate environment
# Note that each example's Project.toml must include Literate as a dependency
using Pkg: Pkg
const EXAMPLEPATH = joinpath(@__DIR__, "..", "examples", EXAMPLEDIR)
@show EXAMPLEPATH
Pkg.activate(EXAMPLEPATH)
Pkg.instantiate()
using Literate: Literate

function preprocess(content)
    # Add link to nbviewer below the first heading of level 1
    sub = SubstitutionString(
        """
#md # ```@meta
#md # EditURL = "@__REPO_ROOT_URL__/examples/@__NAME__/script.jl"
#md # ```
#md #
\\0
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb)
#md #
# *You are seeing the
#md # HTML output generated by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and
#nb # notebook output generated by
# [Literate.jl](https://github.com/fredrikekre/Literate.jl) from the
# [Julia source file](@__REPO_ROOT_URL__/examples/@__NAME__/script.jl).
#md # The corresponding notebook can be viewed in [nbviewer](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb).*
#nb # The rendered HTML can be viewed [in the docs](https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev/examples/@__NAME__/).*
#
        """,
    )
    content = replace(content, r"^# # [^\n]*"m => sub; count=1)

    # remove VSCode `##` block delimiter lines
    content = replace(content, r"^##$."ms => "")

    return content
end

# Convert to markdown and notebook
const SCRIPTJL = joinpath(EXAMPLEPATH, "script.jl")
@show SCRIPTJL
@show OUTDIR
@show EXAMPLESAVENAME
Literate.markdown(
    SCRIPTJL, OUTDIR;
    name=EXAMPLESAVENAME, documenter=false, execute=true, preprocess=preprocess
)
Literate.notebook(
    SCRIPTJL, OUTDIR;
    name=EXAMPLESAVENAME, documenter=false, execute=true, preprocess=preprocess
)

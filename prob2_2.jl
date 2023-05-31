using FlightSims
using ComponentArrays
using LaTeXStrings
using Plots
using ForwardDiff


x1d = t -> 0.0
x1d_dot = t -> ForwardDiff.derivative(x1d, t)
x1d_ddot = t -> ForwardDiff.derivative(x1d_dot, t)
figure_dir = "figures"
mkpath(figure_dir)


struct HWEnv <: AbstractEnv
end


function Dynamics!(env::HWEnv)
    @Loggable function dynamics!(dx, x, a, t; u)
        (; x1, x2) = x
        @log state = x
        @log input = u
        dx.x1 = a*x1 -x1^3 + x2
        dx.x2 = u
    end
end


function State(env::HWEnv)
    function (x1, x2)
        ComponentArray(x1=x1, x2=x2)
    end
end


function backstepping(x, a, t; k1=1, k2=1)
    (; x1, x2) = x
    z1 = x1 - x1d(t)
    x1_dot = a*x1 + x2 - x1^3
    z1_dot = x1_dot - x1d_dot(t)
    x2d = x1^3 - a*x1 + x1d_dot(t) - k1*z1
    x2d_dot = 3*x1^2*x1_dot - a*x1_dot + x1d_ddot(t) - k1*z1_dot
    z2 = x2 - x2d
    u = -k2*z2 + x2d_dot - z1
end


function adaptive_backstepping(x, â, t; k1=1, k2=1, γ=5)
    (; x1, x2) = x
    z1 = x1 - x1d(t)
    x2d = x1^3 - â*x1 + x1d_dot(t) - k1*z1
    z2 = x2 - x2d
    x1_dot_hat = â*x1 - x1^3 + x2
    â_dot = -(1/γ) * (z1*x1 + z2*(3*x1^3 -â*x1 - k1*x1))
    x2d_dot_hat = 3*x1^2*x1_dot_hat - â_dot*x1 - â*x1_dot_hat + x1d_ddot(t) - k1*(x1_dot_hat - x1d_dot(t))
    u = x2d_dot_hat - z1 - k2*z2
    u, â_dot
end


function main_backstepping(x10=2.0, x20=0.0; Δt=0.01, tf=10.0, a=1)
    env = HWEnv()
    x0 = State(env)(x10, x20)
    p0 = a

    @Loggable function dynamics_backstepping!(dx, x, a, t)
        u = backstepping(x, a, t)
        @nested_log Dynamics!(env)(dx, x, a, t; u)
    end

    simulator = Simulator(x0, dynamics_backstepping!, p0; tf=tf)
    df = solve(simulator; savestep=Δt)
    ts = df.time
    x1s = [datum.state.x1 for datum in df.sol]
    x1ds = [x1d(t) for t in df.time]
    x2s = [datum.state.x2 for datum in df.sol]
    us = [datum.input for datum in df.sol]
    fig = plot(layout=(3, 1))
    plot!(
          fig, ts, x1s;
          ylabel=L"$x_{1}$",
          label=nothing,
          subplot=1,
         )
    plot!(
          fig, ts, x1ds;
          ylabel=L"$x_{1_{d}}$",
          subplot=1,
          label=nothing,
          ls=:dash,
          lc=:red,
         )
    plot!(
          fig, ts, x2s;
          ylabel=L"$x_{2}$",
          label=nothing,
          subplot=2,
         )
    plot!(
          fig, ts, us;
          ylabel=L"$u$",
          label=nothing,
          subplot=3,
         )
    savefig(fig, "$(figure_dir)/backstepping.png")
    display(fig)
end


function main_adaptive_backstepping(x10=2.0, x20=0.0; Δt=0.01, tf=10.0, a=1)
    env = HWEnv()
    x0 = State(env)(x10, x20)
    X0 = ComponentArray(system=x0, â=0.0)
    p0 = a

    @Loggable function dynamics_adaptive_backstepping!(dX, X, a, t)
        u, â_dot = adaptive_backstepping(X.system, X.â, t)
        @nested_log Dynamics!(env)(dX.system, X.system, a, t; u)
        @log â = X.â
        dX.â = â_dot
    end

    simulator = Simulator(X0, dynamics_adaptive_backstepping!, p0; tf=tf)
    df = solve(simulator; savestep=Δt)
    ts = df.time
    x1s = [datum.state.x1 for datum in df.sol]
    x1ds = [x1d(t) for t in df.time]
    x2s = [datum.state.x2 for datum in df.sol]
    us = [datum.input for datum in df.sol]
    âs = [datum.â for datum in df.sol]
    fig = plot(layout=(4, 1))
    plot!(
          fig, ts, x1s;
          label=L"$x_{1}$",
          subplot=1,
         )
    plot!(
          fig, ts, x1ds;
          label=L"$x_{1_{d}}$",
          subplot=1,
          ls=:dash,
          lc=:red,
         )
    plot!(
          fig, ts, x2s;
          ylabel=L"$x_{2}$",
          label=nothing,
          subplot=2,
         )
    plot!(
          fig, ts, us;
          ylabel=L"$u$",
          label=nothing,
          subplot=3,
         )
    plot!(
          fig, ts, âs;
          ylabel=L"$\hat{a}$",
          label=nothing,
          subplot=4,
         )
    savefig(fig, "$(figure_dir)/adaptive_backstepping.png")
    display(fig)
end

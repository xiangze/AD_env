# ============================================================================
# Makefile.acados の LQR 追加分(既存 Makefile.acados に追記)
#
# LQR は acados の C ビルド不要(numpy/scipy のみ)。acados-gen をスキップでき、
# 同じ plans/ をそのまま通せる。--controller lqr で切替。
#
# 軽量イメージで回したい場合は Dockerfile.control(scipy だけ)を使う手もあるが、
# 既存の acados イメージにも numpy/scipy は入っているので、そのまま流用できる。
# ============================================================================

# ---- LQR 協調実行(acados-gen 不要) ------------------------------------
cosim-uniad-lqr: up-acados
	$(AC_COMPOSE) --env-file $(ENV_FILE) exec acados \
	  python run_cosim.py --controller lqr \
	    --plan /data/plans/uniad_plan.npz \
	    --out /out/uniad_control_trace_lqr.csv --N $(N) --Tf $(TF) --wheelbase $(WB)

cosim-vad-lqr: up-acados
	$(AC_COMPOSE) --env-file $(ENV_FILE) exec acados \
	  python run_cosim.py --controller lqr \
	    --plan /data/plans/vad_plan.npz \
	    --out /out/vad_control_trace_lqr.csv --N $(N) --Tf $(TF) --wheelbase $(WB)

# ---- 一気通し(UniAD, LQR。export → LQR → report) ----------------------
cosim-uniad-lqr-all: export-plan-uniad up-acados cosim-uniad-lqr cosim-report
	@echo "✓ UniAD → LQR 協調実行 完了(acados ビルド不要)"

# ---- acados と LQR を同じ plan で比較 -----------------------------------
cosim-compare: acados-gen
	$(AC_COMPOSE) --env-file $(ENV_FILE) exec acados bash -c \
	  "python run_cosim.py --controller acados --plan /data/plans/uniad_plan.npz --out /out/cmp_acados.csv && \
	   python run_cosim.py --controller lqr    --plan /data/plans/uniad_plan.npz --out /out/cmp_lqr.csv && \
	   echo '--- acados ---'; tail -n +1 /out/cmp_acados.csv | head -5; \
	   echo '--- lqr ---';    tail -n +1 /out/cmp_lqr.csv    | head -5"

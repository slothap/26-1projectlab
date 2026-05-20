import numpy as np
import random

def calculate_dynamic_cost(t, mu, sigma, D):
    """
    특정 기구의 대기 비용(Cost)을 산출하는 함수
    """
    # 통계적 임계점 계산
    t_70 = mu + 0.524 * sigma  # 누적 확률 70% 지점 (노란색 점등)
    t_95 = mu + 1.645 * sigma  # 누적 확률 95% 지점 (이상치 강등)
    
    # 구간별 wait 계산 (수정된 수식 적용)
    if 0 <= t < t_70:
        # 1구간: 정상 이용 (T_70 기준 잔여 시간)
        t_wait = max((t_70 - t), 1)
    elif t_70 <= t < t_95:
        # 2구간: 종료 임박 (T_95 기준 잔여 시간)
        t_wait = max((t_95 - t), 1)
    else:
        # 3구간: 예상 불가 이상치 (강력한 페널티 부여)
        t_wait = mu * 3 
        
    # 최종 혼잡도 비용 함수
    cost = t_wait + (D * mu)
    return cost

def route_next_equipment(user_routine, equipment_stats):
    """
    개인의 남은 운동 루틴을 바탕으로 역수 비례 가중치를 적용하여 최적 기구 추천
    """
    routine_costs = {}
    inverse_costs_sum = 0.0
    
    # 1. 사용자의 남은 루틴에 속한 기구들의 Cost 산출 및 역수 합계 계산
    for equip_id in user_routine:
        stats = equipment_stats.get(equip_id)
        if not stats:
            continue
            
        cost = calculate_dynamic_cost(
            t=stats['current_t'], 
            mu=stats['mu'], 
            sigma=stats['sigma'], 
            D=stats['demand']
        )
        routine_costs[equip_id] = cost
        inverse_costs_sum += (1.0 / cost)
        
    # 2. 역수 비례 확률 계산
    probabilities = {}
    for equip_id, cost in routine_costs.items():
        prob = (1.0 / cost) / inverse_costs_sum
        probabilities[equip_id] = prob
        
    # 3. 산출된 확률(Prob)을 가중치로 삼아 룰렛 돌리기
    equipments = list(probabilities.keys())
    weights = list(probabilities.values())
    
    recommended_equip = random.choices(equipments, weights=weights, k=1)[0]
    
    return recommended_equip, probabilities, routine_costs

# 랜덤 값 기반 추천 진행
if __name__ == "__main__":
    # 전체 헬스장 기구 목록
    equipment_list = ['bench_press', 'lat_pull_down', 'leg_press', 'squat_rack', 'cable_machine', 'pull_up_bar']
    
    # 기구별 상태 데이터를 랜덤으로 생성
    gym_equipment_status = {}
    for equip in equipment_list:
        mu_val = random.randint(10, 25)              # 평균 이용 시간
        sigma_val = random.randint(2, 6)             # 표준편차
        t_val = random.randint(0, mu_val + 10)       # 현재 사용 시간
        demand_val = random.randint(0, 4)            # 대기 수요
        
        gym_equipment_status[equip] = {
            'current_t': t_val,
            'mu': mu_val,
            'sigma': sigma_val,
            'demand': demand_val
        }
        
    # 생성된 랜덤 값 화면에 출력
    print("=" * 60)
    print("   [ 현재 헬스장 기구 점유 상태 (랜덤 생성) ]")
    print("=" * 60)
    for equip, stats in gym_equipment_status.items():
        print(f" 기구: {equip:15} | 사용(t): {stats['current_t']:2d} | 평균(mu): {stats['mu']:2d} | 편차(sigma): {stats['sigma']:2d} | 대기(D): {stats['demand']:2d}")
    print("-" * 60)
    
    # 사용자의 남은 루틴 (전체 기구 중 3개를 무작위로 뽑기)
    my_remaining_routine = random.sample(equipment_list, 3)
    print(f"\n나의 남은 운동 루틴: {my_remaining_routine}\n")
    
    # 알고리즘 실행
    recommendation, probs, costs = route_next_equipment(my_remaining_routine, gym_equipment_status)
    
    # 결과 출력
    print("--- [ 루틴 내 기구별 Cost 및 추천 확률 분석 ] ---")
    for equip in my_remaining_routine:
        cost = costs[equip]
        prob = probs[equip]
        print(f" [{equip:15}] -> Cost: {cost:6.2f} | 제안 확률: {prob*100:5.1f}%")
        
    print(f"\n>> 시스템 추천 다음 기구: **{recommendation}**\n")
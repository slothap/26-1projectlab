import numpy as np
import random

def calculate_dynamic_cost(t, mu, sigma, D):
    """
    특정 기구의 대기 비용(Cost)을 산출하는 함수
    """
    # 이상치 페널티 시작점: T_off
    t_off = mu + 1.645 * sigma 
    
    # wait 계산
    if t < mu:
        # 정상 구간: 최소 1분 보장
        t_wait = max((mu - t), 1)
    elif mu <= t < t_off:
        # 종료 임박 구간
        t_wait = max((t_off - t), 1)
    else:
        # 예상 불가(이상치 페널티) 구간: 통계적 배제
        t_wait = mu * 3 
        
    # 최종 혼잡도 비용 함수
    cost = t_wait + (D * mu)
    return cost

def route_next_equipment(user_routine, equipment_stats):
    """
    개인의 남은 운동 루틴을 바탕/ 역수 비례 가중치를 적용하여 최적 기구 추천
    """
    routine_costs = {}
    inverse_costs_sum = 0.0
    
    # 1. 사용자의 남은 루틴에 속한 기구들의 Cost 산출 및 역수 합계 계산 0
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
        
    # 3. 가장 높은 확률을 가진 기구를 추천함
    recommended_equip = max(probabilities, key=probabilities.get)
    
    # Cost 값도 출력하기 위해 같이 반환하도록 수정
    return recommended_equip, probabilities, routine_costs

# 랜덤 값 기반 추천 진행
if __name__ == "__main__":
    # 전체 헬스장 기구 목록
    equipment_list = ['bench_press', 'lat_pull_down', 'leg_press', 'squat_rack', 'cable_machine', 'pull_up_bar']
    
    # 기구별 상태 데이터를 랜덤으로 생성
    gym_equipment_status = {}
    for equip in equipment_list:
        mu_val = random.randint(10, 25)              # 평균 이용 시간: 10~25 무작위
        sigma_val = random.randint(2, 6)             # 표준편차: 2~6 무작위
        t_val = random.randint(0, mu_val + 10)       # 현재 사용 시간 (빈 기구부터 예상초과까지 다양하게)
        demand_val = random.randint(0, 4)            # 대기 수요: 0~4명 무작위
        
        gym_equipment_status[equip] = {
            'current_t': t_val,
            'mu': mu_val,
            'sigma': sigma_val,
            'demand': demand_val
        }
        
    # 2. 생성된 랜덤 값 화면에 출력
    print("=" * 60)
    print("   [ 현재 헬스장 기구 점유 상태 (랜덤 생성) ]")
    print("=" * 60)
    for equip, stats in gym_equipment_status.items():
        print(f" 기구: {equip:15} | 사용(t): {stats['current_t']:2d} | 평균(mu): {stats['mu']:2d} | 편차(sigma): {stats['sigma']:2d} | 대기(D): {stats['demand']:2d}")
    print("-" * 60)
    
    # 3.사용자의 남은 루틴 (전체 기구 중 3개를 무작위로 뽑기)
    my_remaining_routine = random.sample(equipment_list, 3)
    print(f"\n나의 남은 운동 루틴: {my_remaining_routine}\n")
    # 4. 알고리즘 실행
    recommendation, probs, costs = route_next_equipment(my_remaining_routine, gym_equipment_status)
    # 5. 결과 출력
    print("--- [ 루틴 내 기구별 Cost 및 추천 확률 분석 ] ---")
    for equip in my_remaining_routine:
        cost = costs[equip]
        prob = probs[equip]
        print(f" [{equip:15}] -> Cost: {cost:6.2f} | 제안 확률: {prob*100:5.1f}%")
        
    print(f"\n>> 시스템 추천 다음 기구: **{recommendation}**\n")